import argparse



# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description="error")

parser.add_argument('--girlfriend', choices=['liud', 'dx'])

args = parser.parse_args()

print(args.girlfriend)


