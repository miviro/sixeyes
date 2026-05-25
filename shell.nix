{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:
pkgs.mkShell {
  packages = [ pkgs.uv pkgs.cudaPackages.cudatoolkit pkgs.arduino-cli ];

  LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib  # libstdc++.so.6
    pkgs.zlib               # libz.so.1
    pkgs.xorg.libxcb        # libxcb.so.1
    pkgs.libGL              # libGL.so.1
    pkgs.glib               # libglib-2.0.so.0, libgthread-2.0.so.0
    pkgs.xorg.libXext       # libXext.so.6
    pkgs.xorg.libX11        # libX11.so.6
    pkgs.xorg.libSM         # libSM.so.6
    pkgs.xorg.libICE        # libICE.so.6
    pkgs.cudaPackages.cudatoolkit        # libcudart.so, libcublas.so, etc.
    pkgs.cudaPackages.cudatoolkit.lib
  ]}:/run/opengl-driver/lib";  # libcuda.so from the NVIDIA driver

  shellHook = ''
    uv sync --quiet
    source .venv/bin/activate

    if ! arduino-cli core list 2>/dev/null | grep -q "esp32:esp32"; then
      arduino-cli config init --overwrite
      arduino-cli core install esp32:esp32
    fi
  '';
}
