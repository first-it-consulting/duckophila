"""Opt-in hook into the microduck_rl body server, so `microduck_rl` itself stays untouched.

On `PYTHONPATH` for the body server only via `duckophila/sim/duck-sim`, and inert unless
`DUCKOPHILA_PROP=1`. Python imports `sitecustomize` at startup, before `duck-sim` runs
`body_server` as `__main__` — so there is no import to hook, but `World.step` calls
`mujoco.mj_step` through the module at call time, and that can be wrapped.

The wrapper, on each physics step of a model that has the prop (`make_scene.py`):
  * moves the prop to the last position it was told (mocap, so the solver just follows), and
  * serves a tiny newline-JSON control port, `DUCKOPHILA_PROP_PORT` (default 7899):
      {"op": "prop", "pos": [x, y, z]}   -> {}            world frame, metres
      {"op": "drum", "pos": [x, y, z], "yaw": rad} -> {}  the striped drum's centre and rotation
      {"op": "where"}                    -> {"tof": {"pos": [...], "xmat": [9 floats]}}
                                            the first duck's ToF site, so a thrower can aim
      {"op": "props"}                    -> {"prop": [x, y, z], "drum": {"pos": [...], "quat": [...]}}
                                            where the ball and the drum are, for a recorder
Any other Python (the system python3 `duck-sim` knocks ports with) finds no mujoco and is untouched.
"""
import os

if os.environ.get('DUCKOPHILA_PROP') == '1':
    try:
        import mujoco
    except ImportError:
        mujoco = None

    if mujoco is not None:
        import json
        import math
        import socketserver
        import threading

        _state = {'pos': None, 'drum': None, 'model': None, 'data': None, 'started': False}
        _real_step = mujoco.mj_step

        class _Handler(socketserver.StreamRequestHandler):
            def handle(self):
                for raw in self.rfile:
                    try:
                        req = json.loads(raw)
                        if req.get('op') == 'prop':
                            _state['pos'] = [float(v) for v in req['pos']]
                            ans = {}
                        elif req.get('op') == 'drum':
                            yaw = float(req.get('yaw', 0.0))
                            _state['drum'] = ([float(v) for v in req['pos']],
                                              [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
                            ans = {}
                        elif req.get('op') == 'props':
                            d = _state['data']
                            ans = {'prop': d.mocap_pos[_state['mocap']].tolist()}
                            if _state.get('drum_mocap', -1) >= 0:
                                k = _state['drum_mocap']
                                ans['drum'] = {'pos': d.mocap_pos[k].tolist(), 'quat': d.mocap_quat[k].tolist()}
                        elif req.get('op') == 'where':
                            m, d = _state['model'], _state['data']
                            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'tof')
                            ans = {'tof': {'pos': d.site_xpos[sid].tolist(), 'xmat': d.site_xmat[sid].tolist()}}
                        else:
                            ans = {'error': f"unknown op {req.get('op')!r}"}
                    except Exception as e:  # a bad request must not take the simulator down
                        ans = {'error': str(e)}
                    self.wfile.write((json.dumps(ans) + '\n').encode())
                    self.wfile.flush()

        class _Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        def _step(model, data, *args, **kw):
            if not _state['started']:
                _state['started'] = True
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'duckophila_prop')
                _state['mocap'] = int(model.body_mocapid[bid]) if bid >= 0 else -1
                did = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'duckophila_drum')
                _state['drum_mocap'] = int(model.body_mocapid[did]) if did >= 0 else -1
                if _state['mocap'] >= 0:
                    _state['model'], _state['data'] = model, data
                    port = int(os.environ.get('DUCKOPHILA_PROP_PORT', '7899'))
                    srv = _Server(('127.0.0.1', port), _Handler)
                    threading.Thread(target=srv.serve_forever, daemon=True).start()
                    print(f'== duckophila prop: control on 127.0.0.1:{port}', flush=True)
                else:
                    print('== duckophila prop: this scene has no prop (use duckophila/sim/duck-sim)', flush=True)
            if data is _state['data']:
                if _state['pos'] is not None:
                    data.mocap_pos[_state['mocap']] = _state['pos']
                if _state['drum'] is not None and _state['drum_mocap'] >= 0:
                    data.mocap_pos[_state['drum_mocap']], data.mocap_quat[_state['drum_mocap']] = _state['drum']
            return _real_step(model, data, *args, **kw)

        mujoco.mj_step = _step
