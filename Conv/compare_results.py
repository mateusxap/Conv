"""
compare_results.py - w_c, v3 (C++) vs PyTorch CPU.

Usage:
    python Conv/compare_results.py <cpp_results.txt> <pytorch_results.txt>

C++ cols (after stripping | and splitting):
    0:hw  1:cin  2:cout  3:wc_c  4:wc_s  5:cw_c  6:cw_s  7:v3_c  8:v3_s ...

PyTorch cols:
    0:hw  1:cin  2:cout  3:pt_c  4:pt_s  5:ratio
"""

import sys


def parse_cpp(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            parts = line.strip().replace("|", " ").split()
            if not parts:
                continue
            try:
                hw   = int(parts[0])
                cin  = int(parts[1])
                cout = int(parts[2])
                wc_c = float(parts[3])
                wc_s = float(parts[4])
                v3_c = float(parts[7])
                v3_s = float(parts[8])
                data[(hw, cin, cout)] = (wc_c, wc_s, v3_c, v3_s)
            except (ValueError, IndexError):
                continue
    return data


def parse_pytorch(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            parts = line.strip().replace("|", " ").split()
            if not parts:
                continue
            try:
                hw   = int(parts[0])
                cin  = int(parts[1])
                cout = int(parts[2])
                pt_c = float(parts[3])
                pt_s = float(parts[4])
                data[(hw, cin, cout)] = (pt_c, pt_s)
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

    W = 140
    print()
    print("=" * W)
    print("  w_c and v3 (C++) vs PyTorch CPU: 1x1 conv, N=1, median time in ms")
    print("  pt/cpp > 1 => C++ faster;  pt/cpp < 1 => PyTorch faster")
    print("=" * W)
    hdr = (
        "{:<5}{:<5}{:<5} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7}"
        " | {:>7} {:>7} | {:>7} {:>7} | {:>6} {:>6}"
    ).format("HW","Cin","Cout","wc_c","wc_s","v3_c","v3_s","pt_c","pt_s",
             "pt/wcc","pt/wcs","pt/v3c","pt/v3s","best_c","best_s")
    print(hdr)
    print("-" * W)

    total = 0
    cnt = {
        "wc": {"c": {"cpp": 0, "PT": 0, "~tie": 0}, "s": {"cpp": 0, "PT": 0, "~tie": 0}},
        "v3": {"c": {"cpp": 0, "PT": 0, "~tie": 0}, "s": {"cpp": 0, "PT": 0, "~tie": 0}},
    }

    for hw in [7, 14, 28, 56]:
        for ci in [16, 32, 64, 128, 256]:
            for co in [16, 32, 64, 128, 256]:
                k = (hw, ci, co)
                if k not in cpp or k not in pt:
                    continue
                total += 1
                wc_c, wc_s, v3_c, v3_s = cpp[k]
                pc, ps = pt[k]

                r_wcc = pc / wc_c if wc_c > 1e-9 else 0.0
                r_wcs = ps / wc_s if wc_s > 1e-9 else 0.0
                r_v3c = pc / v3_c if v3_c > 1e-9 else 0.0
                r_v3s = ps / v3_s if v3_s > 1e-9 else 0.0

                w_wcc = winner(r_wcc);  cnt["wc"]["c"][w_wcc] += 1
                w_wcs = winner(r_wcs);  cnt["wc"]["s"][w_wcs] += 1
                w_v3c = winner(r_v3c);  cnt["v3"]["c"][w_v3c] += 1
                w_v3s = winner(r_v3s);  cnt["v3"]["s"][w_v3s] += 1

                best_c = min((v3_c, "v3"), (wc_c, "wc"), (pc, "PT"))[1]
                best_s = min((v3_s, "v3"), (wc_s, "wc"), (ps, "PT"))[1]

                row = (
                    "{:<5}{:<5}{:<5} | {:7.4f} {:7.4f} | {:7.4f} {:7.4f} | {:7.4f} {:7.4f}"
                    " | {:7.2f} {:7.2f} | {:7.2f} {:7.2f} | {:>6} {:>6}"
                ).format(hw, ci, co, wc_c, wc_s, v3_c, v3_s, pc, ps,
                         r_wcc, r_wcs, r_v3c, r_v3s, best_c, best_s)
                print(row)
        print("-" * W)

    print("\nTotal configs: {}".format(total))
    print("  {:<30} {:>12}   {:>12}".format("", "combined", "sequential"))
    for var, label in [("wc", "w_c vs PyTorch"), ("v3", "v3  vs PyTorch")]:
        cc = cnt[var]["c"]
        cs = cnt[var]["s"]
        print("  {:<30}  cpp:{:3d}  PT:{:3d}  tie:{:3d}   cpp:{:3d}  PT:{:3d}  tie:{:3d}".format(
            label, cc["cpp"], cc["PT"], cc["~tie"], cs["cpp"], cs["PT"], cs["~tie"]))
    print("=" * W)


if __name__ == "__main__":
    main()
