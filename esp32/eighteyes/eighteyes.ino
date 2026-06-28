#include <WiFi.h>
#include <ESPmDNS.h>
#include <WebServer.h>
#include "esp_camera.h"
#include "wifi_config.h"

// ---------------------------------------------------------------------------
// AI Thinker ESP32-CAM pin definitions
// ---------------------------------------------------------------------------
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

#define LED_PIN     33   // active-LOW
#define PAGE_PORT   80   // HTML viewer page
#define STREAM_PORT 81   // MJPEG stream

// ---------------------------------------------------------------------------
// Embedded HTML page (stored in flash to save RAM)
// The stream <img> src is built from window.location.hostname at runtime so
// the page works regardless of what IP the device gets from the router.
// ---------------------------------------------------------------------------
static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head><style>*{margin:0;padding:0}body{background:#000}img{width:100vw;height:100vh;object-fit:contain}</style></head>
<body>
<img id="s">
<script>
var i=document.getElementById('s');
function go(){i.src='http://'+location.hostname+':81/?'+Date.now();}
i.onerror=function(){setTimeout(go,3000);};
go();
</script>
</body>
</html>
)rawliteral";

// ---------------------------------------------------------------------------
// LED helpers
// ---------------------------------------------------------------------------
static void ledBlink(int n, int on_ms = 150, int off_ms = 100) {
    for (int i = 0; i < n; i++) {
        digitalWrite(LED_PIN, LOW);
        delay(on_ms);
        digitalWrite(LED_PIN, HIGH);
        if (i < n - 1) delay(off_ms);
    }
    delay(500);
}

// Blink stage N + 8 rapid blinks, forever — never returns.
static void ledFatal(int stage) {
    for (;;) {
        ledBlink(stage, 300, 200);
        delay(300);
        for (int i = 0; i < 8; i++) {
            digitalWrite(LED_PIN, LOW); delay(40);
            digitalWrite(LED_PIN, HIGH); delay(40);
        }
        delay(1000);
    }
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
bool initCamera() {
    pinMode(PWDN_GPIO_NUM, OUTPUT);
    digitalWrite(PWDN_GPIO_NUM, HIGH); delay(10);
    digitalWrite(PWDN_GPIO_NUM, LOW);  delay(10);

    camera_config_t cfg = {};
    cfg.ledc_channel  = LEDC_CHANNEL_0;
    cfg.ledc_timer    = LEDC_TIMER_0;
    cfg.pin_d0        = Y2_GPIO_NUM;
    cfg.pin_d1        = Y3_GPIO_NUM;
    cfg.pin_d2        = Y4_GPIO_NUM;
    cfg.pin_d3        = Y5_GPIO_NUM;
    cfg.pin_d4        = Y6_GPIO_NUM;
    cfg.pin_d5        = Y7_GPIO_NUM;
    cfg.pin_d6        = Y8_GPIO_NUM;
    cfg.pin_d7        = Y9_GPIO_NUM;
    cfg.pin_xclk      = XCLK_GPIO_NUM;
    cfg.pin_pclk      = PCLK_GPIO_NUM;
    cfg.pin_vsync     = VSYNC_GPIO_NUM;
    cfg.pin_href      = HREF_GPIO_NUM;
    cfg.pin_sscb_sda  = SIOD_GPIO_NUM;
    cfg.pin_sscb_scl  = SIOC_GPIO_NUM;
    cfg.pin_pwdn      = PWDN_GPIO_NUM;
    cfg.pin_reset     = RESET_GPIO_NUM;
    cfg.xclk_freq_hz  = 20000000;
    cfg.pixel_format  = PIXFORMAT_JPEG;
    cfg.frame_size    = CAM_FRAME_SIZE;
    cfg.jpeg_quality  = CAM_JPEG_QUALITY;
    cfg.fb_count      = psramFound() ? 2 : 1;
    cfg.grab_mode     = CAMERA_GRAB_LATEST;  // always freshest frame
    cfg.fb_location   = CAMERA_FB_IN_PSRAM;

    if (esp_camera_init(&cfg) != ESP_OK) return false;

    sensor_t* s = esp_camera_sensor_get();
    if (s) { s->set_vflip(s, 0); s->set_hmirror(s, 0); }
    return true;
}

// ---------------------------------------------------------------------------
// HTTP handlers (port 80)
// ---------------------------------------------------------------------------
WebServer webServer(PAGE_PORT);

void handleRoot() {
    webServer.send_P(200, "text/html", INDEX_HTML);
}

// ---------------------------------------------------------------------------
// MJPEG stream (port 81)
// Reads frames continuously and pushes them as multipart/x-mixed-replace.
// Blocks until the client disconnects — only one viewer at a time.
// ---------------------------------------------------------------------------
WiFiServer streamServer(STREAM_PORT);

static void runMjpegStream(WiFiClient& client) {
    // Send HTTP response headers
    client.print(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
        "Pragma: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "\r\n"
    );

    char hdr[96];
    while (client.connected()) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) break;

        snprintf(hdr, sizeof(hdr),
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %u\r\n"
            "\r\n",
            (unsigned)fb->len);
        client.print(hdr);
        size_t sent = client.write(fb->buf, fb->len);
        esp_camera_fb_return(fb);

        if (sent == 0) break;
        client.print("\r\n");
    }
}

// Drain incoming HTTP request headers up to the blank line
static void drainHttpHeaders(WiFiClient& client) {
    String line;
    uint32_t t0 = millis();
    while (client.connected() && millis() - t0 < 2000) {
        if (!client.available()) { delay(1); continue; }
        char c = client.read();
        if (c == '\n') {
            if (line.length() <= 1) return;  // blank line = end of headers
            line = "";
        } else if (c != '\r') {
            line += c;
        }
    }
}

// ---------------------------------------------------------------------------
// setup / loop
// ---------------------------------------------------------------------------
int deviceNum = 1;  // assigned during setup via mDNS scan

void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);

    // Proof-of-life: 5 rapid blinks.
    // If these don't appear, setup() never ran (global-constructor crash).
    for (int i = 0; i < 5; i++) {
        digitalWrite(LED_PIN, LOW); delay(80);
        digitalWrite(LED_PIN, HIGH); delay(80);
    }
    delay(500);

    // Stage 1: WiFi station connect (20 s timeout)
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    {
        uint32_t t0 = millis();
        while (WiFi.status() != WL_CONNECTED) {
            if (millis() - t0 > 20000) ledFatal(1);
            delay(200);
        }
    }
    // Scan for existing eighteyesN devices and claim the lowest unused number.
    MDNS.begin("eighteyes_probe");
    delay(200);
    {
        bool used[16] = {};
        int found = MDNS.queryService("http", "tcp");
        for (int i = 0; i < found; i++) {
            String h = MDNS.hostname(i);
            if (h.startsWith("eighteyes")) {
                int num = h.substring(9).toInt();
                if (num >= 1 && num <= 16) used[num - 1] = true;
            }
        }
        while (deviceNum <= 16 && used[deviceNum - 1]) deviceNum++;
    }
    MDNS.end();
    {
        String mdnsName = "eighteyes" + String(deviceNum);
        MDNS.begin(mdnsName.c_str());
    }
    MDNS.addService("http", "tcp", PAGE_PORT);
    MDNS.addService("http", "tcp", STREAM_PORT);
    ledBlink(1);

    // Stage 2: camera
    if (!initCamera()) ledFatal(2);
    ledBlink(2);

    // Start servers
    webServer.on("/", handleRoot);
    webServer.begin();
    streamServer.begin();

    // Boot complete
    ledBlink(10, 60, 60);
}

void loop() {
    // Reconnect WiFi if the link drops between stream sessions
    if (WiFi.status() != WL_CONNECTED) {
        WiFi.reconnect();
        uint32_t t0 = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - t0 < 10000)
            delay(200);
    }

    webServer.handleClient();

    // Blink deviceNum times every 5 s while idle to identify this device.
    static uint32_t lastBlink = 0;
    if (millis() - lastBlink > 5000) {
        ledBlink(deviceNum, 200, 150);
        lastBlink = millis();
    }

    // Accept at most one MJPEG stream client at a time.
    // While streaming, the web server (port 80) is unresponsive — the browser
    // already has the page loaded before the stream starts, so this is fine.
    WiFiClient client = streamServer.available();
    if (!client) return;

    drainHttpHeaders(client);
    digitalWrite(LED_PIN, LOW);
    runMjpegStream(client);
    digitalWrite(LED_PIN, HIGH);
    client.stop();
}
