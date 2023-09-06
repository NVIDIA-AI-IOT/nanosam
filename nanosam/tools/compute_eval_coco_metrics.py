import json
import argparse


def filter_results_by_area(results, min=None, max=None):
    filtered = []
    for r in results:
        if min is not None and r['area'] < min:
            continue
        if max is not None and r['area'] > max:
            continue
        filtered.append(r)
    return filtered


def filter_results_by_category_id(results, category_id):
    return [r for r in results if r['category_id'] == category_id]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("coco_results", type=str, default="data/mobile_sam_coco_results.json")
    parser.add_argument("--category_id", type=int, default=None)
    parser.add_argument("--size", type=str, default="all", choices=["all", "small", "medium", "large"])
    args = parser.parse_args()

    print(args)

    with open(args.coco_results, 'r') as f:
        results = json.load(f)

    if args.size == "small":
        results = filter_results_by_area(results, None, 32**2)
    elif args.size == "medium":
        results = filter_results_by_area(results, 32**2, 96**2)
    elif args.size == "large":
        results = filter_results_by_area(results, 96**2, None)

    if args.category_id is not None:
        results = filter_results_by_category_id(results, args.category_id)

    miou = sum(r['iou'] for r in results) / len(results)

    print(f"mIOU: {miou}")