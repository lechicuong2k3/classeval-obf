from src.utils.obfuscator import obfuscate_code
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="file to original code", default="original.py")
parser.add_argument("--output", help="file to output obfuscated code", default="obfuscated.py")
parser.add_argument("--scheme", help="scheme to use", default=1)
parser.add_argument("--name_length", help="name length", default="medium")
args = parser.parse_args()
def main():
    with open(args.file, "r") as f:
        code = f.read()
    obfuscated_code,_ = obfuscate_code(code, scheme=args.scheme, name_length = args.name_length)
    with open(args.output, "w") as f:
        f.write(obfuscated_code)
    print(f"Obfuscated code saved to {args.output}")


if __name__ == "__main__":
    main()