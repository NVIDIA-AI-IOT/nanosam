import json
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def compute_metrics(results):
    miou_nanosam_coco = sum(r['iou_nanosam_coco'] for r in results) / len(results)
    miou_mobilesam_coco = sum(r['iou_mobilesam_coco'] for r in results) / len(results)
    miou_mobilesam_nanosam = sum(r['iou_mobilesam_nanosam'] for r in results) / len(results)
    return {
        "miou_nanosam_coco": miou_nanosam_coco,
        "miou_mobilesam_coco": miou_mobilesam_coco,
        "miou_mobilesam_nanosam": miou_mobilesam_nanosam
    }



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


with open("eval_coco_results.json", 'r') as f:
    results = json.load(f)

results_small = filter_results_by_area(results, None, 32**2)
results_medium = filter_results_by_area(results, 32**2, 96**2)
results_large = filter_results_by_area(results, 96**2, None)
results_person = filter_results_by_category_id(results, 1) # person
results_bicycle = filter_results_by_category_id(results, 2) # bicycle
results_car = filter_results_by_category_id(results, 3) # car
results_chair = filter_results_by_category_id(results, 62) # chair

print(len(results))
metrics = {
    "all": compute_metrics(results),
    "size_small": compute_metrics(results),
    "size_medium": compute_metrics(results_medium),
    "size_large": compute_metrics(results_large),
    "category_person": compute_metrics(results_person),
    "category_bicycle": compute_metrics(results_bicycle),
    "category_car": compute_metrics(results_car),
    "category_chair": compute_metrics(results_chair)
}

with open("coco_eval_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)


with open("coco_eval_metrics.csv", 'w') as f:
    f.write(f"Subset, NanoSAM vs. COCO, MobileSAM vs. COCO, MobileSAM vs. NanoSAM\n")
    for name, data in metrics.items():
        f.write(f"{name}, {data['miou_nanosam_coco']}, {data['miou_mobilesam_coco']}, {data['miou_mobilesam_nanosam']}\n")
