import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch

torch.backends.cudnn.benchmark = True 

from utils.utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, AverageMeter, LoadImages,
)
from adas.visualizer import ADASVisualizer

def make_parser():
    p = argparse.ArgumentParser(description='YOLOPv2 ADAS Perception Demo')
    p.add_argument('--weights', type=str, default='data/weights/yolopv2.pt')
    p.add_argument('--source',  type=str, default='data/example.jpg')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--conf-thres', type=float, default=0.3)
    p.add_argument('--iou-thres', type=float, default=0.45)
    p.add_argument('--device', default='0')
    p.add_argument('--nosave', action='store_true')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--project', default='runs/detect')
    p.add_argument('--name', default='adas')
    p.add_argument('--exist-ok', action='store_true')
    return p

def detect():
    global opt
    source, weights = opt.source, opt.weights
    save_img = not opt.nosave and not source.endswith('.txt')
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    wp = Path(weights)
    if not wp.is_file():
        raise FileNotFoundError(f"Weights not found: '{wp.resolve()}'.")

    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = torch.jit.load(str(wp), map_location=device)
    model = model.to(device)
    if half: model.half()
    model.eval()

    adas = ADASVisualizer(img_h=720, img_w=1280)
    dataset = LoadImages(source, img_size=opt.img_size, stride=32)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters())))

    vid_path, vid_writer = None, None
    fps, frame_count, fps_start = 0.0, 0, time.time()

    print(f'\n  ADAS Demo  |  Source: {source}  |  Device: {device}')
    print(f'  Press Q or ESC to quit.\n')

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3: img = img.unsqueeze(0)

        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres)

        da_mask = driving_area_mask(seg)
        ll_mask = lane_line_mask(ll)

        for i, det in enumerate(pred):
            im0 = im0s.copy()
            det_np = det.cpu().numpy() if len(det) else np.zeros((0, 6))

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            results = adas.process_frame(im0, da_mask, ll_mask, det_np)
            rendered = adas.render(im0, results, t2 - t1, fps, str(device).upper())

            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed > 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            if not opt.no_display:
                cv2.imshow('ADAS Perception — YOLOPv2', rendered)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27): return

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(str(save_dir / Path(path).name), rendered)
                else:
                    sp = str(save_dir / Path(path).name)
                    if vid_path != sp:
                        vid_path = sp
                        if isinstance(vid_writer, cv2.VideoWriter): vid_writer.release()
                        f = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                        if not f or f <= 0: f = 30
                        h, w = rendered.shape[:2]
                        vid_writer = cv2.VideoWriter(sp + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), f, (w, h))
                    vid_writer.write(rendered)

    if isinstance(vid_writer, cv2.VideoWriter): vid_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)
    with torch.no_grad():
        detect()