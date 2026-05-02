import os
import re

def _rename():
    logs_dir = "logs_3"
    pattern = re.compile(r"^(ep\d+)-loss[\d.]+-val_loss[\d.]+\.pth$")

    for filename in os.listdir(logs_dir):
        match = pattern.match(filename)
        if match:
            new_name = f"{match.group(1)}.pth"
            src = os.path.join(logs_dir, filename)
            dst = os.path.join(logs_dir, new_name)
            os.rename(src, dst)
            print(f"{filename} -> {new_name}")

