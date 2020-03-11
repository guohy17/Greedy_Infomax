from optparse import OptionGroup

def parse_GIM_args(parser):
    group = OptionGroup(parser, "Greedy InfoMax training options")
    group.add_option(
        "--learning_rate", type="float", default=2e-4, help="Learning rate"
    )
    group.add_option(
        "--prediction_step",
        type="int",
        default=5,
        help="Time steps to predict into future",
    )
    group.add_option(
        "--negative_samples",
        type="int",
        default=16,
        help="Number of negative samples to be used for training",
    )

    parser.add_option_group(group)
    return parser
