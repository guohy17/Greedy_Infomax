from optparse import OptionGroup

def parser_reload_args(parser):
    group = OptionGroup(parser, "Reloading pretrained model options")

    ### Options to load pretrained models
    group.add_option(
        "--start_epoch",
        type="int",
        default=0,
        help="Epoch to start GIM training from: "
        "v=0 - start training from scratch, "
        "v>0 - load pre-trained model that was trained for v epochs and continue training "
        "(path to pre-trained model needs to be specified in opt.model_path)",
    )
    group.add_option(
        "--model_path",
        type="string",
        default="./logs/module2",
        help="Directory of the saved model (path within --data_input_dir)",
    )
    group.add_option(
        "--model_num",
        type="string",
        default="99",
        help="Number of the saved model to be used for training the linear classifier"
        "(loaded using model_path + model_X.ckpt, where X is the model_num passed here)",
    )
    parser.add_option_group(group)
    return parser