import numpy as np
import cv2
import json

output_file = 'output.csv'
param_file = 'param.json'

cap = cv2.VideoCapture(2)

ret, frame = cap.read()
assert ret

# cap.set(cv2.CAP_PROP_SETTINGS, 1)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)
# cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
cap.set(cv2.CAP_PROP_FOCUS, 100)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow('frame')


def nothing(x):
    pass


line_num = 8
line_names = []
for l in range(0, line_num):
    line_name = str(l)
    cv2.createTrackbar(line_name, 'frame',
                       (height//line_num)*l, height, nothing)
    line_names.append(line_name)

# threshold等の値はデバイスによって変わる
cv2.createTrackbar('threshold', 'frame', 150, 255, nothing)


def set_exposure(value):
    re = cap.set(cv2.CAP_PROP_EXPOSURE, -10+value)
    print(re)


cv2.createTrackbar('exposure', 'frame', 5, 8, set_exposure)


def set_focus(value):
    re = cap.set(cv2.CAP_PROP_FOCUS, 100+value*5)
    print(re)


cv2.createTrackbar('focus', 'frame', 0, 120, set_focus)

if output_file is not None:
    with open(output_file, 'w') as f:
        f.write('theta,residue,points')
    file = None

    def change_save_mode(value):
        global file
        if value == 1 and file is None:
            file = open(output_file, 'a')
        elif value == 0 and file is not None:
            file.close()
            file = None
    cv2.createTrackbar('save mode', 'frame', 0, 1, change_save_mode)

if param_file is not None:
    with open(param_file) as f:
        param_dict = json.load(f)
    for key, value in param_dict.items():
        cv2.setTrackbarPos(key, 'frame', value)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # パラメータの値を取得
    line_height = [cv2.getTrackbarPos(line_name, 'frame')
                   for line_name in line_names]
    threshold = cv2.getTrackbarPos('threshold', 'frame')

    # 明るい点のマスクを作成
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold

    frame[mask] = [0, 255, 0]

    # 区切り線の描画
    for h, name in zip(line_height, line_names):
        frame = cv2.line(frame, (0, h), (width, h), (255, 0, 0), 5)
        frame = cv2.putText(
            frame, name, (0, h+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 各区間で明るい点の重心を計算
    bright_pos = []
    for l in range(len(line_height)-1):
        upper = line_height[l]
        lower = line_height[l+1]
        bright = mask[upper:lower, :]
        if bright.any() and lower-upper > 0:
            x_pos = np.repeat(np.arange(0, width)[
                              None, :], lower-upper, axis=0)
            bright_cm = x_pos[bright].sum()/bright.sum()
            bright_pos.append((bright_cm, (upper+lower)/2))

    # 明るい点が２つ以上あるときに線形回帰
    if len(bright_pos) >= 2:
        x = [pos[0] for pos in bright_pos]
        y = [pos[1] for pos in bright_pos]

        # x = ay + b
        A = np.vstack([y, np.ones(len(y))]).T
        p, res, rnk, s = np.linalg.lstsq(A, x, rcond=None)
        a, b = p

        # 角度を得る
        theta = np.abs(np.arctan(a))
        if res.shape[0] == 0:
            res = 0
        else:
            res = res[0] / (len(bright_pos)-2)

        if output_file is not None:
            if file is not None:
                file.write(f'\n{theta},{res},{len(bright_pos)}')

        if b < 0:
            p0 = (0, int(-b/a))
        elif width <= b:
            p0 = (width-1, int((width-1-b)/a))
        else:
            p0 = (int(b), 0)

        x1 = a * height + b
        if x1 < 0:
            p1 = (0, int(-b/a))
        elif width <= x1:
            p1 = (width-1, int((width-1-b)/a))
        else:
            p1 = (int(x1), height-1)
        frame = cv2.line(frame, p0, p1, (0, 255, 255), 5)
        frame = cv2.putText(frame, '{:.2f} degrees'.format(np.degrees(
            theta)), (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    for posx, posy in bright_pos:
        frame = cv2.circle(frame, (int(posx), int(posy)), 5, (0, 0, 255), -1)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        param_dict = {}
        for h, name in zip(line_height, line_names):
            param_dict[name] = h
        param_dict['threshold'] = threshold
        param_dict['exposure'] = cv2.getTrackbarPos('exposure', 'frame')
        param_dict['focus'] = cv2.getTrackbarPos('focus', 'frame')
        with open('param.json', 'w') as f:
            json.dump(param_dict, f, indent=4)


if output_file is not None:
    if file is not None:
        file.close()

cap.release()
cv2.destroyAllWindows()
