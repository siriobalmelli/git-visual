with import <nixpkgs>{};

let
  pyenv = python3.withPackages(p: with p; [
    matplotlib
  ]);

in
  mkShell {
    # 'with python3Packages' clobbers systemd in pkgconfig ?!?
    buildInputs = [
      less  # used by python help()

      pyenv
    ];
  }
