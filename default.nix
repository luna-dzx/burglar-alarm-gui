{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python313Packages.pygame
    pkgs.python313Packages.numpy
    pkgs.python313Packages.opencv-python
    pkgs.python313Packages.pyserial
    pkgs.python313Packages.imutils
    pkgs.python313
   ];
}
