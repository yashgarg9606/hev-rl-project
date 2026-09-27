"""Convert an explicitly supplied NEDC velocity file; run --help for inputs."""

if __package__:
    from .velocity_cycle_input import export_from_cli
else:
    from velocity_cycle_input import export_from_cli


def main():
    export_from_cli('NEDC', 'V_nedc.txt', 1180, 120.0)


if __name__ == "__main__":
    main()
