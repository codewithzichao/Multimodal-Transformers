import argparse

def MyArgs(model_name="xlm-roberta", base_path="/Users/codewithzichao/Desktop/competitions/meme_EACL2021", \
           epochs=50, batch_size=8, max_norm=0.25, accum_num=8, fold_num=5):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=base_path, help="please input base path", required=False)
    parser.add_argument("--model_name", type=str, default=model_name, help="please input model name", required=False)
    parser.add_argument("--epochs", type=int, default=epochs, help="please input epochs", required=False)
    parser.add_argument("--batch_size", type=int, default=batch_size, help="please input batch size", required=False)
    parser.add_argument("--max_norm", type=float, default=max_norm, help="please input max_norm", required=False)
    parser.add_argument("--fold_num", type=int, default=fold_num, help="please input fold num", required=False)
    parser.add_argument("--accum_num", type=int, default=accum_num, help="please input accum num", required=False)

    args = parser.parse_args()

    return args
