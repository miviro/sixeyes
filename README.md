# SixEyes

# Webcam resolution
The Creative Live! CAM Sync 1080P V2 can capture different resolutions at different fps:
```
v4l2-ctl --list-formats-ext -d /dev/video0 
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'MJPG' (Motion-JPEG, compressed)
		Size: Discrete 1920x1080
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 544x288
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 432x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 800x600
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 864x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 960x720
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1024x576
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.033s (30.000 fps)
	[1]: 'YUYV' (YUYV 4:2:2)
		Size: Discrete 1920x1080
			Interval: Discrete 0.200s (5.000 fps)
		Size: Discrete 640x480
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 544x288
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 320x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 432x240
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 160x120
			Interval: Discrete 0.033s (30.000 fps)
		Size: Discrete 800x600
			Interval: Discrete 0.050s (20.000 fps)
		Size: Discrete 864x480
			Interval: Discrete 0.050s (20.000 fps)
		Size: Discrete 960x720
			Interval: Discrete 0.080s (12.500 fps)
		Size: Discrete 1024x576
			Interval: Discrete 0.067s (15.000 fps)
		Size: Discrete 1280x720
			Interval: Discrete 0.100s (10.000 fps)
```

For the sake of YOLOv11, we will go for 30fps, so that leaves us with 640x480@30 as the best choice.

# Latency justification
Since we want to operate at 30fps, that leaves us with a margin of 1/30s=33.33ms per frame.

On average, on the desktop PC (RTX 3070 Ti), we have these timings:
```
0: 480x640 1 person, 8.2ms
Speed: 0.6ms preprocess, 8.2ms inference, 1.5ms postprocess per image at shape (1, 3, 480, 640)
```
Of which we only care about the *8.2ms* inference.
Then, to send a coordinate via serial, neglecting processing time:
```
3 bytes (0xFF, 0xPT, 0xYW) / 115200baud@1s8d1s = 260us
```
Let's consider it 1ms to account for kernel/controller latency.
