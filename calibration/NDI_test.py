# -*- coding: utf-8 -*-


import NDIlib as ndi, time, numpy as np, cv2

assert ndi.initialize()
finder = ndi.find_create_v2(); time.sleep(5)
sources = ndi.find_get_current_sources(finder) or []
print("Sources:", [s.ndi_name for s in sources])
src = sources[0]; ndi.find_destroy(finder)

rc = ndi.RecvCreateV3()
rc.color_format = ndi.RECV_COLOR_FORMAT_FASTEST
rc.bandwidth = ndi.RECV_BANDWIDTH_HIGHEST
rc.create.allow_video_fields = False 
recv = ndi.recv_create_v3(rc.create)

miss = 0; hits = 0
while hits < 30:
    t, v, a, m = ndi.recv_capture_v2(recv, 1000)
    if t == ndi.FRAME_TYPE_VIDEO:
        img = np.frombuffer(v.data, dtype=np.uint8).reshape(v.yres, v.xres, 4).copy()
        ndi.recv_free_video_v2(recv, v)
        hits += 1
        if hits % 10 == 0:
            print("Got", hits, "frames")
    else:
        miss += 1
        if miss % 5 == 0:
            print("No video yet")
ndi.recv_destroy(recv); ndi.destroy()
