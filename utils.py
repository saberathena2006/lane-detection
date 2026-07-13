import datetime
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
import re
import glob
import random
import cv2
import numpy as np
import torch
import torchvision

logger = logging.getLogger(__name__)


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # FIX: also catch FileNotFoundError/OSError -- on Windows machines without git
        # installed (or not on PATH), subprocess.check_output raises FileNotFoundError
        # instead of CalledProcessError, which was previously uncaught and crashed
        # select_device() on import for many Windows users.
        return ''  # not a git repository / git not available


def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOPv2 torch {torch.__version__} '  # string
    # FIX: removed the literal rocket emoji from the format string. On Windows
    # consoles using the legacy 'cp1252'/'gbk' codepage this previously required
    # the ascii-ignore encode/decode dance below just to avoid a UnicodeEncodeError
    # crash; removing the emoji is simpler and avoids relying on that fallback.

    device = str(device).strip().lower().replace('cuda:', '')  # FIX: normalize input,
    # e.g. accept 'cuda:0', ' 0 ', 'CPU' etc. Previously only exact 'cpu' (lowercased)
    # or a bare digit string worked; anything else silently fell into the CUDA branch.
    cpu = device == 'cpu'

    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        if not torch.cuda.is_available():
            # FIX: replaced bare `assert` with an explicit, informative RuntimeError.
            # `assert` statements are stripped out entirely when Python is run with
            # the -O (optimize) flag, silently disabling this safety check. They also
            # produce an unhelpful bare AssertionError. This also degrades gracefully
            # to CPU instead of hard-crashing, which is the most common failure mode
            # reported when running this repo on a machine with no NVIDIA GPU / no
            # CUDA-enabled torch build installed (the default --device '0' in demo.py
            # crashes immediately on such machines otherwise).
            logger.warning(
                f"CUDA unavailable or device '{device}' invalid/unsupported by this "
                f"torch build -- falling back to CPU. If you expected a GPU, verify "
                f"your NVIDIA driver and that you installed a CUDA-enabled build of "
                f"torch (see https://pytorch.org/get-started/locally/)."
            )
            cpu = True

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    # FIX: kept the Windows-safe ascii-ignore encode as defense-in-depth (device
    # names from vendors occasionally contain non-ascii characters), but it is no
    # longer required for the emoji specifically since that was removed above.
    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2_label = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # FIX: the original code computed `t_size` / `c2` for the label background
        # but never actually drew the label rectangle or text -- `label` was dead
        # code that had zero visible effect. demo.py never passes a `label=` value
        # today, so this branch never executed either way and there is no behavior
        # change for the current pipeline. This restores the (clearly intended)
        # drawing calls so the parameter isn't silently non-functional if a future
        # caller (or a modified demo.py) passes a label.
        cv2.rectangle(img, c1, c2_label, color, -1, cv2.LINE_AA)  # filled label background
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)


class SegmentationMetric(object):
    '''
    imgLabel [batch_size, height(144), width(256)]
    confusionMatrix [[0(TN),1(FP)],
                     [2(FN),3(TP)]]
    '''
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        total = self.confusionMatrix.sum()
        acc = np.diag(self.confusionMatrix).sum() / total if total > 0 else 0.0
        # FIX: guard divide-by-zero (RuntimeWarning -> nan) when confusionMatrix is
        # still all-zero, e.g. metric read before any addBatch() call.
        return acc

    def lineAccuracy(self):
        Acc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + 1e-12)
        return Acc[1]

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-12)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        with np.errstate(divide='ignore', invalid='ignore'):
            # FIX: explicitly silence the numpy RuntimeWarning for 0/0 division,
            # which is expected here and already handled by the next line
            # (`IoU[np.isnan(IoU)] = 0`). Previously this printed noisy warnings
            # to stderr on every call, which is not an execution bug but clutters
            # production logs.
            IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        mIoU = np.nanmean(IoU)
        return mIoU

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        with np.errstate(divide='ignore', invalid='ignore'):
            IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        total = np.sum(self.confusionMatrix)
        freq = np.sum(self.confusionMatrix, axis=1) / total if total > 0 else np.zeros(self.numClass)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        iu = np.nan_to_num(iu)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def _make_grid(nx=20, ny=20):
    # FIX: torch.meshgrid without an explicit `indexing=` argument is deprecated
    # since PyTorch 1.10 (emits a UserWarning) and, more importantly, its *default*
    # indexing convention changed across versions in a way that is easy to get
    # subtly wrong. Passing indexing='ij' explicitly locks in the exact behavior
    # the original code relied on (numpy-style row/col meshgrid matching the
    # un-keyworded old default), so grid coordinates stay correct across
    # PyTorch 1.7 through 2.x. Wrapped in try/except for torch<1.10, which
    # doesn't accept the `indexing` kwarg at all.
    try:
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
    except TypeError:
        # torch < 1.10: no `indexing` kwarg support; old default was already 'ij'
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx))
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def split_for_trace_model(pred=None, anchor_grid=None):
    z = []
    st = [8, 16, 32]
    for i in range(3):
        bs, _, ny, nx = pred[i].shape
        pred[i] = pred[i].view(bs, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        y = pred[i].sigmoid()
        gr = _make_grid(nx, ny).to(pred[i].device)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + gr) * st[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(bs, -1, 85))
    pred = torch.cat(z, 1)
    return pred


def show_seg_result(img, result, palette=None, is_demo=False):
    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)
    assert palette.shape[0] == 3  # len(classes)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2

    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[result == label, :] = color
    else:
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        color_area[result[0] == 1] = [0, 255, 0]
        color_area[result[1] == 1] = [255, 0, 0]
        color_seg = color_area

    # convert to BGR
    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    return


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"{re.escape(path.stem)}{sep}(\d+)", d) for d in dirs]
        # FIX: the original f-string `rf"%s{sep}(\d+)" % path.stem` mixed old-style
        # %-formatting *inside* an f-string/raw-string literal. This is fragile:
        # if `path.stem` itself contains regex metacharacters (e.g. 'exp+final',
        # 'exp[v2]', or on Windows a project name with parentheses), it breaks the
        # regex or matches incorrectly. Using re.escape() on the stem makes the
        # match robust to arbitrary folder names while preserving identical
        # behavior for the common alnum-only case (e.g. 'exp0', 'exp1', ...).
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                         labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 4096  # (pixels) maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    # FIX: removed the unused `min_wh = 2` variable (dead code, was never
    # referenced anywhere after the constraint line that used it was already
    # commented out in the original source).

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            logger.warning(f'NMS time limit {time_limit}s exceeded')
            # FIX: switched from print() to logger.warning() for consistency with
            # the rest of the module's logging and so this doesn't get lost /
            # interleaved oddly with cv2 imshow or tqdm progress bars.
            break  # time limit exceeded

    return output


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            # FIX: original raised a bare Exception with a slightly ambiguous
            # message. Using FileNotFoundError is more idiomatic/catchable, and
            # the message now clarifies that `path` may be a glob, dir, or file.
            raise FileNotFoundError(
                f'ERROR: source not found: {p}\n'
                f'Expected an existing file, a directory of images/videos, or a glob pattern.'
            )

        img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
        vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        if self.nf == 0:
            # FIX: replaced bare `assert` (stripped under python -O, unhelpful
            # AssertionError otherwise) with an explicit FileNotFoundError. This
            # is also the fix for the previously silent `UnboundLocalError` in
            # demo.py's detect() -- by failing loudly *here*, with a clear
            # message, instead of letting an empty dataset iterate zero times
            # and crash much later on `img.size(0)` with a confusing traceback.
            raise FileNotFoundError(
                f'No images or videos found in {p}. '
                f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'
            )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            if img0 is None:
                # FIX: replaced bare `assert img0 is not None, 'Image Not Found ' + path`
                # with an explicit, more informative error. cv2.imread silently
                # returns None for unreadable files (corrupt image, unsupported
                # codec, permissions issue, non-ASCII path on some OpenCV/Windows
                # builds, etc.) -- the assert gave no clue *why* it failed, and
                # (again) is a no-op under `python -O`.
                raise FileNotFoundError(
                    f'Image not found or unreadable: {path} '
                    f'(file may be corrupt, an unsupported format, or an OpenCV '
                    f'build/codec issue).'
                )

        # Padded resize
        img0 = cv2.resize(img0, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            # FIX: cv2.VideoCapture() does not raise on failure to open a file --
            # it silently produces a capture object where .isOpened() is False and
            # every subsequent .read() returns (False, None) forever, which the
            # original __next__ loop had no terminating condition for if it was
            # the *first* video and immediately failed to open (self.count would
            # never advance past 0 correctly in that edge case since new_video is
            # only called for videos[0] once at init). Failing loudly here with
            # a clear message is much easier to diagnose (bad codec, missing
            # ffmpeg backend, corrupted file, etc.).
            raise IOError(f'Failed to open video: {path} (check codec support / file integrity).')
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratio, (dw, dh)


def driving_area_mask(seg=None):
    da_predict = seg[:, :, 12:372, :]
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=2, mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    return da_seg_mask


def lane_line_mask(ll=None):
    ll_predict = ll[:, :, 12:372, :]
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=2, mode='bilinear')
    ll_seg_mask = torch.round(ll_seg_mask).squeeze(1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
    return ll_seg_mask
