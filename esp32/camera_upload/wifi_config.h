#pragma once

// WiFi station credentials — must match your access point
#define WIFI_SSID     ""
#define WIFI_PASSWORD ""

// Camera capture settings
// Resolution options (OV2640 max is UXGA 1600×1200, but lower = higher FPS):
//   FRAMESIZE_UXGA  1600×1200  ~0.5–1 fps over WiFi
//   FRAMESIZE_SXGA  1280×1024  ~1–2 fps
//   FRAMESIZE_XGA   1024×768   ~2–4 fps
//   FRAMESIZE_SVGA   800×600   ~5–10 fps  ← recommended for streaming
//   FRAMESIZE_VGA    640×480   ~8–15 fps
#define CAM_FRAME_SIZE   FRAMESIZE_VGA
#define CAM_JPEG_QUALITY 10  // 0–63, lower = better quality / larger file
