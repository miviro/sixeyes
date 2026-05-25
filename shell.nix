{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [ pkgs.uv ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib  # libstdc++.so.6
    pkgs.zlib               # libz.so.1
    pkgs.xorg.libxcb        # libxcb.so.1
    pkgs.libGL              # libGL.so.1
    pkgs.glib               # libglib-2.0.so.0, libgthread-2.0.so.0
    pkgs.xorg.libXext       # libXext.so.6
    pkgs.xorg.libX11        # libX11.so.6
    pkgs.xorg.libSM         # libSM.so.6
    pkgs.xorg.libICE        # libICE.so.6
  ];
}
