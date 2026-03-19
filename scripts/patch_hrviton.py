import re

def patch_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    fixed = []
    for line in lines:
        if re.match(r'\s*if opt\.cuda\s*:', line):
            indent = len(line) - len(line.lstrip())
            fixed.append(' ' * indent + 'if False:  # cuda disabled\n')
        elif line.strip().endswith('.cuda()') and '=' not in line:
            indent = len(line) - len(line.lstrip())
            fixed.append(' ' * indent + 'pass  # cuda disabled\n')
        elif re.search(r'(\w+)\s*=\s*(\w+)\.cuda\(\)', line):
            fixed.append(re.sub(r'(\w+)\s*=\s*(\w+)\.cuda\(\)', r'\1 = \2', line))
        else:
            fixed.append(line)
    with open(filepath, 'w') as f:
        f.writelines(fixed)
    print(f"Patched: {filepath}")

if __name__ == '__main__':
    patch_file('HR-VITON/networks.py')
    patch_file('HR-VITON/test_generator.py')

    with open('HR-VITON/test_generator.py', 'r') as f:
        content = f.read()
    content = content.replace('np.float)', 'float)')
    content = content.replace('np.float,', 'float,')
    with open('HR-VITON/test_generator.py', 'w') as f:
        f.write(content)

    with open('HR-VITON/network_generator.py', 'r') as f:
        content = f.read()
    content = content.replace('torch.randn(b, w, h, 1).cuda()', 'torch.randn(b, w, h, 1)')
    with open('HR-VITON/network_generator.py', 'w') as f:
        f.write(content)

    print("All patches applied ✓")
