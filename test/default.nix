with import <nixpkgs> {};

let thisaixi = pkgs.lib.overrideDerivation mcaixictw (attrs: {
      name = "custom-aixi";
      src  = ./..;
    });
in stdenv.mkDerivation {
     name = "mc-aixi-ctw-tests";

     src = ./.;

     buildInputs = [thisaixi
                    pkgs.haskellPackages.ghc
                    pkgs.haskellPackages.QuickCheck];
   }
