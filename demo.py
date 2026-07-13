import argparse
import time
from pathlib import Path
import cv2
import torch

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized, select_device, increment_path, \
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, \
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, \
    AverageMeter, \
    LoadImages


def make_parser():
    parser = argparse.ArgumentParser()
    # FIX: changed `type=str, default='...'` combined with `nargs='+'` to just
    # `type=str` (no nargs). With `nargs='+'`, argparse ALWAYS returns a *list*,
    # even for the default value -- so `opt.weights` was `['data/weights/yolopv2.pt']`,
    # a list, not a string. `torch.jit.load()` requires a path-like/string/file
    # object and does not accept a list, so the very first line of detect() that
    # touches weights (`torch.jit.load(weights)`) crashed unconditionally, on
    # every run, regardless of whether the file existed. This is almost
    # certainly the root cause of the failure. `nargs='+'` made sense for the
    # upstream YOLOv5-style repos that support ensembling multiple weight files,
    # but this script's detect() never handles a list of models, so preserving
    # `nargs='+'` while fixing the crash would require restructuring the whole
    # inference pipeline for a feature that isn't implemented -- out of scope.
    parser.add_argument('--weights', type=str, default='data/weights/yolopv2.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # FIX: changed default from '0' to '' (empty string). The old default '0'
    # forced CUDA device 0 and made select_device() hard-crash via `assert
    # torch.cuda.is_available()` on any machine without a CUDA GPU / without a
    # CUDA-enabled torch build -- which is the overwhelmingly common case for
    # people just trying to run a demo. An empty string now lets
    # utils.select_device() auto-detect: it will use CUDA if available and
    # transparently fall back to CPU (with a warning) otherwise. Explicitly
    # passing --device 0 or --device cpu still works exactly as before.
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser


def detect():
    # setting and directories
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # FIX: validate the weights path exists *before* attempting to load it, and
    # raise a clear, actionable error instead of letting torch.jit.load() throw
    # an opaque internal error (e.g. "PytorchStreamReader failed reading zip
    # archive" or a raw OSError) when the path is wrong/missing. This directly
    # addresses the "put yolopv2.pt in weights" scenario -- if it's saved under
    # a different filename or subfolder than data/weights/yolopv2.pt, this now
    # tells you exactly what path was checked instead of crashing deep inside
    # a torch internals with a cryptic traceback.
    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at '{weights_path.resolve()}'. "
            f"Pass the correct path with --weights, e.g. "
            f"--weights data/weights/yolopv2.pt"
        )

    # FIX (critical, order-of-operations bug): select_device() must run BEFORE
    # torch.jit.load(). The original code called torch.jit.load(weights) FIRST,
    # then select_device() afterward. torch.jit.load() with no `map_location`
    # deserializes tensors onto whatever device they were *saved* on. If the
    # TorchScript model was traced/saved on a CUDA machine and you run this demo
    # on a CPU-only machine (no NVIDIA GPU, or torch installed without CUDA
    # support), torch.jit.load(weights) crashes immediately trying to allocate
    # CUDA memory that doesn't exist -- before select_device() ever gets a
    # chance to decide CPU vs GPU. Reordering these calls, and passing an
    # explicit map_location, makes weight loading robust regardless of which
    # device the checkpoint was originally saved on.
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    stride = 32
    try:
        model = torch.jit.load(str(weights_path), map_location=device)
        # FIX: explicit map_location=device (see above). Also cast weights_path
        # to str explicitly -- torch.jit.load has historically been inconsistent
        # about accepting pathlib.Path across versions/platforms, so passing a
        # plain string is the safest cross-version choice.
    except RuntimeError as e:
        # FIX: wrap the load call so a corrupted/incompatible checkpoint file
        # produces a clear, actionable message instead of a raw torch
        # RuntimeError with a stack trace that doesn't mention the file at all.
        raise RuntimeError(
            f"Failed to load TorchScript weights from '{weights_path}'. "
            f"The file may be corrupted, incomplete, or not a valid TorchScript "
            f"(.pt) archive produced by torch.jit.save/torch.jit.trace/script. "
            f"Original error: {e}"
        ) from e

    model = model.to(device)

    if half:
        model.half()  # to FP16
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # NOTE: LoadImages now raises FileNotFoundError immediately if `source`
    # resolves to zero images/videos (see utils.py fix), instead of silently
    # constructing an empty dataset that would iterate zero times below and
    # crash much later with a confusing UnboundLocalError on `img`/`t2`/etc.

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # FIX: initialize timing variables before the loop so that if, for any
    # future reason, the loop body were to exit early, referencing these names
    # afterward fails predictably rather than with an UnboundLocalError deep in
    # unrelated code. (With the LoadImages fix above guaranteeing at least one
    # item, this is now purely defensive, but costs nothing and removes a class
    # of "works on my machine" failures for anyone who relaxes that check later.)
    t1 = t2 = t3 = t4 = tw1 = tw2 = time.time()
    img = None

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of torch.jit.trace causes extra time
        # consumption in demo version but this problem will not appear in
        # official version
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                    agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            if not fps or fps <= 0:
                                # FIX: some containers/codecs report FPS as 0 or
                                # NaN via CAP_PROP_FPS (notably certain webcam
                                # streams / malformed mp4 headers). Passing 0 to
                                # cv2.VideoWriter silently produces a writer that
                                # either fails to open or writes an unplayable
                                # 0-fps file with no error raised. Fall back to a
                                # sane default so output videos are always
                                # playable.
                                fps = 30
                            w, h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        if not vid_writer.isOpened():
                            # FIX: cv2.VideoWriter does not raise on failure to
                            # open (e.g. missing codec backend, invalid output
                            # extension). It silently no-ops on every .write()
                            # call after, producing a 0-byte or missing output
                            # file with zero error messages. Raising here turns
                            # a silent, hard-to-diagnose failure into an
                            # immediate, clear one.
                            raise IOError(
                                f"Failed to open video writer for '{save_path}'. "
                                f"This is usually caused by a missing 'mp4v' "
                                f"codec backend in your OpenCV build. Try "
                                f"installing 'opencv-python' with FFMPEG support, "
                                f"or change the output extension/codec."
                            )
                    vid_writer.write(im0)

    if isinstance(img, torch.Tensor):
        # FIX: guard against `img` never having been assigned (would previously
        # raise UnboundLocalError here if the dataset loop body never executed).
        # With the LoadImages fix guaranteeing >=1 item this branch is always
        # taken in practice, but this keeps the function robust even if that
        # invariant is ever relaxed.
        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))
        print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')

    if isinstance(vid_writer, cv2.VideoWriter):
        # FIX: the original code never released the final VideoWriter after the
        # main loop finished -- it only released the *previous* writer when
        # switching to a new video mid-loop. This means the very last output
        # video's file buffer was never flushed/closed, frequently producing a
        # truncated or unplayable .mp4 file (moov atom never written) depending
        # on timing and OS file-handle cleanup.
        vid_writer.release()


if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
