"""
compare_results.py - w_c, v3 (C++) vs PyTorch CPU.

Usage:
    python Conv/compare_results.py <cpp_results.txt> <pytorch_results.txt>

C++ cols (after stripping | and splitting):
    0:hw  1:cin  2:cout  3:kh  4:kw  5:wc_c  6:wc_s  7:v3_c  8:v3_s ...

PyTorch cols:
    0:hw  1:cin  2:cout  3:kh  4:kw  5:pt_c  6:pt_s  7:ratio
"""

import sys


def parse_cpp(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            parts = line.strip().replace("|", " ").split()
            if len(parts) < 9:
                continue
            try:
                hw   = int(parts[0])
                cin  = int(parts[1])
                cout = int(parts[2])
                kh   = int(parts[3])
                kw   = int(parts[4])
                wc_c = float(parts[5])
                wc_s = float(parts[6])
                v3_c = float(parts[7])
                v3_s = float(parts[8])
                data[(hw, cin, cout, kh, kw)] = (wc_c, wc_s, v3_c, v3_s)
            except (ValueError, IndexError):
                continue
    return data


def parse_pytorch(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            parts = line.strip().replace("|", " ").split()
            if len(parts) < 7:
                continue
            try:
                hw   = int(parts[0])
                cin  = int(parts[1])
                cout = int(parts[2])
                kh   = int(parts[3])
                kw   = int(parts[4])
                pt_c = float(parts[5])
                pt_s = float(parts[6])
                data[(hw, cin, cout, kh, kw)] = (pt_c, pt_s)
            except (ValueError, IndexError):
                continue
    return data


def winner(ratio, tol=0.05):
    if ratio > 1 + tol:
        return "cpp"
    if ratio < 1 - tol:
        return "PT"
    return "~tie"


def main():
    if len(sys.argv) != 3:
        print("Usage: python Conv/compare_results.py <cpp_results.txt> <pytorch_results.txt>")
        sys.exit(1)

    cpp = parse_cpp(sys.argv[1])
    pt  = parse_pytorch(sys.argv[2])

    W = 148
    print()
    print("=" * W)
    print("  w_c and v3 (C++) vs PyTorch CPU: valid conv, N=1, median time in ms")
    print("  pt/cpp > 1 => C++ faster;  pt/cpp < 1 => PyTorch faster")
    print("=" * W)
    hdr = "{:<5}{:<5}{:<5}{:<4}{:<4} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>6} {:>6}".format(
        "HW","Cin","Cout","KH","KW",
        "wc_c","wc_s","v3_c","v3_s","pt_c","pt_s",
        "pt/wcc","pt/wcs","pt/v3c","pt/v3s","best_c","best_s")
    print(hdr)
    print("-" * W)

    total = 0
    cnt = {
        "wc": {"c": {"win":0,"PT":0,"~tie":0}, "s": {"win":0,"PT":0,"~tie":0}},
        "v3": {"c": {"win":0,"PT":0,"~tie":0}, "s": {"win":0,"PT":0,"~tie":0}},
    }

    HW_VALS  = [7, 14, 28, 56]
    CIN_VALS = [16, 32, 64, 128, 256]
    KERNEL_CONFIGS = [
        (1, 1, [16, 32, 64, 128, 256],  [16, 32, 64, 128, 256, 512, 1024]),
        (3, 3, [16, 32, 64, 128, 256],  [512, 1024]),
        (3, 3, [128, 512],              [48]),
    ]

    for kh, kw, cin_vals, cout_vals in KERNEL_CONFIGS:
        for hw in HW_VALS:
            for ci in cin_vals:
                for co in cout_vals:
                    k = (hw, ci, co, kh, kw)
                    if k not in cpp or k not in pt:
                        continue
                    total += 1
                    wc_c, wc_s, v3_c, v3_s = cpp[k]
                    pc, ps = pt[k]

                    r_wcc = pc / wc_c if wc_c > 1e-9 else 0.0
                    r_wcs = ps / wc_s if wc_s > 1e-9 else 0.0
                    r_v3c = pc / v3_c if v3_c > 1e-9 else 0.0
                    r_v3s = ps / v3_s if v3_s > 1e-9 else 0.0

                    w_wcc = winner(r_wcc).replace("cpp","win");  cnt["wc"]["c"][w_wcc] += 1
                    w_wcs = winner(r_wcs).replace("cpp","win");  cnt["wc"]["s"][w_wcs] += 1
                    w_v3c = winner(r_v3c).replace("cpp","win");  cnt["v3"]["c"][w_v3c] += 1
                    w_v3s = winner(r_v3s).replace("cpp","win");  cnt["v3"]["s"][w_v3s] += 1

                    best_c = min((v3_c,"v3"),(wc_c,"wc"),(pc,"PT"))[1]
                    best_s = min((v3_s,"v3"),(wc_s,"wc"),(ps,"PT"))[1]

                    row = "{:<5}{:<5}{:<5}{:<4}{:<4} | {:7.4f} {:7.4f} | {:7.4f} {:7.4f} | {:7.4f} {:7.4f} | {:7.2f} {:7.2f} | {:7.2f} {:7.2f} | {:>6} {:>6}".format(
                        hw, ci, co, kh, kw,
                        wc_c, wc_s, v3_c, v3_s, pc, ps,
                        r_wcc, r_wcs, r_v3c, r_v3s, best_c, best_s)
                    print(row)
            print("-" * W)

    print("\nTotal configs: {}".format(total))
    print("  {:<30} {:>12}   {:>12}".format("", "combined", "sequential"))
    for var, label in [("wc","w_c vs PyTorch"), ("v3","v3  vs PyTorch")]:
        cc = cnt[var]["c"]; cs = cnt[var]["s"]
        wins_label = var  # "wc" or "v3"
        print("  {:<30}  {}:{:3d}  PT:{:3d}  tie:{:3d}   {}:{:3d}  PT:{:3d}  tie:{:3d}".format(
            label, wins_label, cc["win"], cc["PT"], cc["~tie"], wins_label, cs["win"], cs["PT"], cs["~tie"]))
    print("=" * W)


if __name__ == "__main__":
    main()
