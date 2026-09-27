"""Convert an explicitly supplied WLTC Class3b velocity file; run --help for inputs."""

if __package__:
    from .velocity_cycle_input import export_from_cli
else:
    from velocity_cycle_input import export_from_cli


def main():
    export_from_cli('WLTC Class3b', 'V_class3b.txt', 1801, 140.0)


if __name__ == "__main__":
    main()
