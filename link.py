"""Reaching a duck over WebRTC — the brain on a laptop, the duck on the LAN.

On the robot itself the bridge uses the daemons' unix sockets (`robotd`, `tofd`, `mediad`'s
`media.frame`). Anywhere else it takes what the robot publishes for a consumer, which is a
WebRTC session (microduck `docs/design/remote-webrtc.md`): `mediad`'s signalling server on port
8443, an H.264 video track, and a `control` datachannel that pipes JSON-RPC to the same daemons
(`robot.subscribe`, `tof.stream`, `robot.move`, `robot.head`, ... are all routed). One session
carries everything, so `RtcLink` hands out channel objects with the unix-socket `Rpc`'s interface
(`send`, `lines`) and sorts what comes back: a reply by its id, to whoever asked; a notification
by its method, to whoever subscribed. `mediad` forwards replies uncorrelated, on purpose.

**When a frame was taken.** The unix socket stamps every frame (`captured_at_unix_us`); a WebRTC
frame carries only an RTP timestamp, from a random origin. What dates it is the RTCP sender report
— an RTP timestamp and the wall clock at the same instant, the robot's wall clock — and
`media.video`, which reads the robot's monotonic and wall clocks together, puts that on the
clock `robot.state` and `tof.frame` stamp with (`t_ns`). So a frame here gets a capture time on
the robot's own monotonic clock, the state it pairs with is found by `t_ns`, and the network's
delay drops out of the pairing altogether. Until the first sender report arrives, a frame has no
capture time and the walking efference copy waits.

    .venv/bin/python -I link.py ws://<duck>:8443     # a probe: frames, rate, state and depth
"""
from __future__ import annotations

import asyncio
import itertools
import json
import queue
import threading
import time

import numpy as np

RTP_CLOCK = 90000        # H.264's RTP clock rate
NTP_UNIX = 2208988800    # seconds from 1900 to 1970


class Channel:
    """One client of the session, with the unix-socket `Rpc`'s interface."""

    def __init__(self, link: 'RtcLink'):
        self.link = link
        self.inbox: queue.Queue = queue.Queue()

    def send(self, method: str, params=None, request: bool = False):
        self.link.send(self, method, params, request)

    def lines(self):
        while True:
            yield self.inbox.get()


class RtcLink:
    def __init__(self, url: str, timeout: float = 20.0):
        self.url = url
        self.ids = itertools.count(1)
        self.owner: dict[int, Channel] = {}             # request id -> who asked
        self.subscribers: dict[str, list[Channel]] = {}  # notification method -> who subscribed
        self.rotate = 0
        self.clock_offset_ns: int | None = None          # robot CLOCK_REALTIME - CLOCK_MONOTONIC
        self.sr: tuple[float, int] | None = None         # last sender report: (unix s, RTP timestamp)
        self.frame = None                                # (grey upright, capture robot-mono s or None, arrival)
        self.frame_seq = 0
        self.fresh = threading.Condition()
        self._buf = ''
        self._dc = None
        self._ready = threading.Event()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self._task = asyncio.run_coroutine_threadsafe(self._connect(), self.loop)
        if not self._ready.wait(timeout):
            if self._task.done() and self._task.exception():
                raise self._task.exception()
            raise TimeoutError(f'no WebRTC control channel from {url} within {timeout:.0f} s')
        self._video_info()

    # ── sync API ───────────────────────────────────────────────────────────────────────────────
    def channel(self) -> Channel:
        return Channel(self)

    def send(self, ch: Channel, method: str, params=None, request: bool = False):
        msg = {'jsonrpc': '2.0', 'method': method}
        if params is not None:
            msg['params'] = params
        if request:
            msg['id'] = next(self.ids)
            self.owner[msg['id']] = ch
        if method == 'robot.subscribe':
            self.subscribers.setdefault('robot.state', []).append(ch)
        elif method == 'tof.stream':
            self.subscribers.setdefault('tof.frame', []).append(ch)
        line = json.dumps(msg) + '\n'
        self.loop.call_soon_threadsafe(self._dc.send, line)

    def next_frame(self, timeout: float = 2.0):
        """The next new frame: (upright grey image, capture time on the robot's monotonic clock in
        seconds — or None before the first sender report —, arrival time here)."""
        with self.fresh:
            seq = self.frame_seq
            if not self.fresh.wait_for(lambda: self.frame_seq != seq, timeout):
                raise TimeoutError('no video frame')
            return self.frame

    # ── inside the loop ────────────────────────────────────────────────────────────────────────
    async def _connect(self):
        import websockets
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from aiortc.sdp import candidate_from_sdp

        self.pc = pc = RTCPeerConnection()
        ws = await websockets.connect(self.url, max_size=None)
        session = None

        @pc.on('track')
        def on_track(track):
            if track.kind == 'video':
                self._watch_sender_reports(pc)
                asyncio.ensure_future(self._pull(track))

        @pc.on('datachannel')
        def on_dc(ch):
            if ch.label != 'control':
                return
            self._dc = ch
            ch.on('message', self._on_message)
            self._ready.set()

        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get('type')
            if kind == 'welcome':
                await ws.send(json.dumps({'type': 'list'}))
            elif kind == 'list' and session is None and msg.get('producers'):
                await ws.send(json.dumps({'type': 'startSession', 'peerId': msg['producers'][0]['id']}))
            elif kind == 'sessionStarted':
                session = msg['sessionId']
            elif kind == 'peer' and 'sdp' in msg:
                await pc.setRemoteDescription(RTCSessionDescription(**msg['sdp']))
                await pc.setLocalDescription(await pc.createAnswer())
                await ws.send(json.dumps({'type': 'peer', 'sessionId': session,
                                          'sdp': {'type': 'answer', 'sdp': pc.localDescription.sdp}}))
            elif kind == 'peer' and msg.get('ice', {}).get('candidate'):
                c = msg['ice']['candidate']
                cand = candidate_from_sdp(c.split(':', 1)[1] if c.startswith('candidate:') else c)
                cand.sdpMLineIndex = msg['ice'].get('sdpMLineIndex')
                await pc.addIceCandidate(cand)
            elif kind == 'error':
                print(f'webrtc signalling: {msg.get("details")}', flush=True)

    def _watch_sender_reports(self, pc):
        """Keep the newest RTCP sender report of the video stream: wall clock against RTP time."""
        from aiortc.rtp import RtcpSrPacket
        for receiver in pc.getReceivers():
            if receiver.track is None or receiver.track.kind != 'video':
                continue
            real = receiver._handle_rtcp_packet

            async def handle(packet, real=real):
                if isinstance(packet, RtcpSrPacket):
                    ntp = packet.sender_info.ntp_timestamp
                    self.sr = ((ntp >> 32) + (ntp & 0xFFFFFFFF) / 2 ** 32 - NTP_UNIX, packet.sender_info.rtp_timestamp)
                return await real(packet)
            receiver._handle_rtcp_packet = handle
            self._receiver = receiver

    def _capture_mono(self, pts: int) -> float | None:
        """A frame's capture time on the robot's monotonic clock, seconds."""
        if self.sr is None or self.clock_offset_ns is None:
            return None
        origin = getattr(self._receiver, '_RTCRtpReceiver__timestamp_mapper')._origin
        if origin is None:
            return None
        rtp = (pts + origin) & 0xFFFFFFFF
        d = ((rtp - self.sr[1] + 2 ** 31) & 0xFFFFFFFF) - 2 ** 31    # signed, wrap-safe
        unix = self.sr[0] + d / RTP_CLOCK
        return unix - self.clock_offset_ns / 1e9

    async def _pull(self, track):
        while True:
            try:
                f = await track.recv()
            except Exception:
                return
            y = f.to_ndarray(format='gray')
            grey = np.ascontiguousarray(np.rot90(y, k=-self.rotate // 90))
            with self.fresh:
                self.frame = (grey, self._capture_mono(f.pts), time.time())
                self.frame_seq += 1
                self.fresh.notify_all()

    def _on_message(self, data):
        """Datachannel messages are neither one object each nor line-aligned: a large reply comes in
        pieces, and objects need not end in a newline. So decode objects off the front of a buffer."""
        self._buf += data if isinstance(data, str) else data.decode()
        dec = json.JSONDecoder()
        while True:
            self._buf = self._buf.lstrip()
            if not self._buf:
                return
            try:
                msg, end = dec.raw_decode(self._buf)
            except ValueError:
                return                                   # incomplete: wait for the rest
            self._buf = self._buf[end:]
            if not isinstance(msg, dict):
                continue
            if 'id' in msg and msg['id'] in self.owner:
                self.owner.pop(msg['id']).inbox.put(msg)
            elif 'method' in msg:
                for ch in self.subscribers.get(msg['method'], ()):
                    ch.inbox.put(msg)

    def _video_info(self):
        ch = self.channel()
        ch.send('media.video', request=True)
        msg = next(ch.lines())
        r = msg.get('result') or {}
        self.rotate = int(r.get('rotate', 0))
        if r.get('real_ns') and r.get('mono_ns'):
            self.clock_offset_ns = int(r['real_ns']) - int(r['mono_ns'])
        else:
            print('media.video gave no clocks: frames will have no capture time (older mediad)', flush=True)


if __name__ == '__main__':
    import sys
    link = RtcLink(sys.argv[1] if len(sys.argv) > 1 else 'ws://127.0.0.1:8443')
    print(f'connected: rotate {link.rotate}, clocks {"yes" if link.clock_offset_ns is not None else "no"}', flush=True)
    st, tof = link.channel(), link.channel()
    st.send('robot.subscribe', {'hz': 50}, request=True)
    tof.send('tof.stream', request=True)
    last_state = {}

    def read(ch, key):
        for m in ch.lines():
            if m.get('method'):
                last_state[key] = (m['params'].get('t_ns', 0) / 1e9, time.time())
    threading.Thread(target=read, args=(st, 'state'), daemon=True).start()
    threading.Thread(target=read, args=(tof, 'tof'), daemon=True).start()
    n, t0, lags = 0, time.time(), []
    while time.time() - t0 < 10:
        grey, cap, arr = link.next_frame()
        n += 1
        if cap is not None and 'state' in last_state:
            s_mono, s_arr = last_state['state']
            lags.append((s_mono - cap) - (s_arr - arr))     # how much older the frame is than the state, beyond arrival
    print(f'{n / 10:.1f} fps, {grey.shape} upright; dated frames {len(lags)}/{n}; '
          f'state {"yes" if "state" in last_state else "no"}, depth {"yes" if "tof" in last_state else "no"}')
    if lags:
        print(f'video arrives {np.median(lags) * 1000:+.0f} ms behind the state stream (median; capture to display)')
