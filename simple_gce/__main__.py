from simple_gce import parse_cli_args

if __name__ == "__main__":
    parser = parse_cli_args()
    args = parser.parse_args()
    args.func(args)
