with import <nixpkgs> {};

stdenv.mkDerivation {
  name = "mc-aixi-ctw";

  src = ./.;

  buildInputs = [];
}
