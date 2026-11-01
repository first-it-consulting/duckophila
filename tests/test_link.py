"""The WebRTC link's bookkeeping, without a peer: message framing, routing, and dating frames."""
import json
import queue

import pytest

pytest.importorskip('aiortc')
from link import Channel, RtcLink, RTP_CLOCK


def bare():
    link = RtcLink.__new__(RtcLink)
    link._buf, link.owner, link.subscribers = '', {}, {}
    return link


def test_fragments_and_missing_newlines_are_reassembled_and_routed():
    link = bare()
    asker, sub = Channel(link), Channel(link)
    link.owner[7] = asker
    link.subscribers['robot.state'] = [sub]
    reply = json.dumps({'jsonrpc': '2.0', 'id': 7, 'result': {'ok': True}})
    note = json.dumps({'jsonrpc': '2.0', 'method': 'robot.state', 'params': {'t_ns': 1}})
    stream = reply + note + '\n'                       # two objects, no newline between them
    for piece in (stream[:10], stream[10:57], stream[57:]):
        link._on_message(piece)
    assert asker.inbox.get_nowait()['result'] == {'ok': True}
    assert sub.inbox.get_nowait()['params']['t_ns'] == 1
    with pytest.raises(queue.Empty):
        asker.inbox.get_nowait()                       # a reply goes once, to whoever asked


def test_frame_date_from_sender_report_across_rtp_wrap():
    class Mapper:
        _origin = 2 ** 32 - 1000                      # the stream started just before the wrap
    class Receiver:
        _RTCRtpReceiver__timestamp_mapper = Mapper()
    link = bare()
    link._receiver = Receiver()
    link.clock_offset_ns = 1_000 * 10 ** 9            # robot wall clock = monotonic + 1000 s
    link.sr = (5000.0, 2 ** 32 - 1000)                # the report: RTP origin at unix 5000 s
    pts = 1000 + RTP_CLOCK                            # one second and the wrap later
    assert link._capture_mono(pts) == pytest.approx(5000.0 + 1000 / RTP_CLOCK + 1.0 - 1000.0)
