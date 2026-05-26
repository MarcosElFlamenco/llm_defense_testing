from utils.eval_utils import build_arg_parser, run_autodan_eval


if __name__ == "__main__":
    args = build_arg_parser(include_iter=False).parse_args()
    run_autodan_eval(args, attack_mode="ga")