import numpy as np
import cv2
import json
import os
import time

output_dir = 'output'
movie_out = 'test.mp4'
param_file = 'param.json'
layer_num = 8 # 層の数

if output_dir is not None:
    os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(1)

ret, frame = cap.read()
assert ret

# cap.set(cv2.CAP_PROP_SETTINGS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
#cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)
# cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
#cap.set(cv2.CAP_PROP_FOCUS, 100)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow('raw_frame', cv2.WINDOW_AUTOSIZE)

### 切り取る枠を設定
frame_points = [[int(width/4),int(height/4)], [int(width*3/4),int(height/4)], [int(width*3/4),int(height*3/4)], [int(width/4),int(height*3/4)]]
mapping_width = int(width/2)
mapping_height = int(int(height/2))
mapping_points = [[0,0], [mapping_width,0], [mapping_width,mapping_height], [0,mapping_height]]
point_frame_active = True
def set_frame_points(event, x, y, flags,param):
    global frame_points, mapping_points, mapping_width, mapping_height, point_frame_active
    if event == cv2.EVENT_LBUTTONDOWN and point_frame_active:
        min_distance = 1e6
        min_distance_index = -1
        for i, (fp_x,fp_y) in enumerate(frame_points):
            distance = np.sqrt((x-fp_x)**2 + (y-fp_y)**2)
            if distance < min_distance:
                min_distance = distance
                min_distance_index = i
        frame_points[min_distance_index] = [x, y]

        fp_x0, fp_y0 = frame_points[0]
        fp_x1, fp_y1 = frame_points[1]
        mapping_width = int(np.sqrt((fp_x1-fp_x0)**2 + (fp_y1-fp_y0)**2))
        fp_x0, fp_y0 = frame_points[0]
        fp_x1, fp_y1 = frame_points[3]
        mapping_height = int(np.sqrt((fp_x1-fp_x0)**2 + (fp_y1-fp_y0)**2))
        mapping_points = [[0,0], [mapping_width,0], [mapping_width,mapping_height], [0,mapping_height]]
        print(mapping_height, mapping_width)
cv2.setMouseCallback('raw_frame', set_frame_points)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = frame.copy()
    for i in range(4):
        fp_x, fp_y = frame_points[i]
        raw_frame = cv2.circle(raw_frame, (fp_x, fp_y), 5, (0, 255, 0), -1)
        raw_frame = cv2.putText(raw_frame, str(i+1), (fp_x, fp_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        raw_frame = cv2.line(raw_frame, tuple(frame_points[i]), tuple(frame_points[(i+1)%4]), (0, 255, 255), 2)
    cv2.imshow('raw_frame', raw_frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('n'):
        point_frame_active = False
        break
pts1 = np.float32(frame_points)
pts2 = np.float32(mapping_points)
M = cv2.getPerspectiveTransform(pts1,pts2)
###


cv2.namedWindow('controller', cv2.WINDOW_NORMAL)
cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

def nothing(x):
    pass

line_num = layer_num + 1
line_names = []
for l in range(0, line_num):
    line_name = str(l)
    cv2.createTrackbar(line_name, 'controller',
                       (mapping_height//line_num)*l, mapping_height, nothing)
    line_names.append(line_name)

# threshold等の値はデバイスによって変わる
cv2.createTrackbar('threshold', 'controller', 150, 255, nothing)


def set_exposure(value):
    re = cap.set(cv2.CAP_PROP_EXPOSURE, -10+value)
    print(re)
cv2.createTrackbar('exposure', 'controller', 5, 8, set_exposure)


def set_focus(value):
    re = cap.set(cv2.CAP_PROP_FOCUS, 100+value*5)
    print(re)
cv2.createTrackbar('focus', 'controller', 0, 120, set_focus)

filter = 1
def set_filter(value):
    global filter
    filter = value*2 + 1
cv2.createTrackbar('filter', 'controller', 0, 30, set_filter)

count = 0
st = 0
if output_dir is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    movie = None
    save_image = False
    fps = 15

    def change_save_mode(value):
        global movie,count, st, save_image
        if value == 1:
            movie = cv2.VideoWriter(output_dir+'/'+movie_out,fourcc, fps, (mapping_width,mapping_height))
            count = 0
            st = time.time()
            save_image = True
        elif value == 0:
            movie.release()
            movie = None
            save_image = False
    cv2.createTrackbar('save mode', 'controller', 0, 1, change_save_mode)

if param_file is not None:
    with open(param_file) as f:
        param_dict = json.load(f)
    for key, value in param_dict.items():
        cv2.setTrackbarPos(key, 'controller', value)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = frame.copy()
    for i in range(4):
        fp_x, fp_y = frame_points[i]
        raw_frame = cv2.circle(raw_frame, (fp_x, fp_y), 5, (0, 255, 0), -1)
        raw_frame = cv2.putText(raw_frame, str(i+1), (fp_x, fp_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        raw_frame = cv2.line(raw_frame, tuple(frame_points[i]), tuple(frame_points[(i+1)%4]), (0, 255, 255), 2)
    cv2.imshow('raw_frame', raw_frame)
    frame = cv2.warpPerspective(frame, M, (mapping_width,mapping_height))
    raw = frame.copy()

    # パラメータの値を取得
    line_height = [cv2.getTrackbarPos(line_name, 'controller')
                   for line_name in line_names]
    threshold = cv2.getTrackbarPos('threshold', 'controller')

    # 明るい点のマスクを作成
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, filter)
    mask = gray > threshold

    frame[mask] = [0, 255, 0]
    masked = frame.copy()

    # 区切り線の描画
    for h, name in zip(line_height, line_names):
        frame = cv2.line(frame, (0, h), (mapping_width, h), (255, 0, 0), 5)
        frame = cv2.putText(
            frame, name, (0, h+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    lined = frame.copy()

    # 各区間で明るい点の重心を計算
    bright_pos = []
    bright_x = []
    bright_y = []
    bright_x_std = []
    bright_y_err = []
    for l in range(layer_num):
        upper = line_height[l]
        lower = line_height[l+1]
        bright = mask[upper:lower, :]
        if bright.any() and lower-upper > 0:
            x_pos = np.repeat(np.arange(0, mapping_width)[
                              None, :], lower-upper, axis=0)
            bright_x_cm = x_pos[bright].mean()
            bright_y_cm = (upper+lower)/2
            bright_pos.append((bright_x_cm, bright_y_cm))
            bright_x.append(bright_x_cm)
            bright_x_std.append(x_pos[bright].std())
            bright_y.append(bright_y_cm)
            bright_y_err.append((lower-upper)/2)
        else:
            bright_x.append('')
            bright_x_std.append('')
            bright_y.append('')
            bright_y_err.append('')
    
    for posx, posy in bright_pos:
        frame = cv2.circle(frame, (int(posx), int(posy)), 5, (0, 0, 255), -1)
    points = frame.copy()


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

        if b < 0:
            p0 = (0, int(-b/a))
        elif mapping_width <= b:
            p0 = (mapping_width-1, int((mapping_width-1-b)/a))
        else:
            p0 = (int(b), 0)

        x1 = a * mapping_height + b
        if x1 < 0:
            p1 = (0, int(-b/a))
        elif mapping_width <= x1:
            p1 = (mapping_width-1, int((mapping_width-1-b)/a))
        else:
            p1 = (int(x1), mapping_height-1)
        frame = cv2.line(frame, p0, p1, (0, 255, 255), 5)
        display_pos = (int((p0[0] + p1[0]) / 2), int((p0[1] + p1[1]) / 2))
        frame = cv2.putText(frame, '{:.2f} degrees'.format(np.degrees(
            theta)), display_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        detect = frame.copy()
        if output_dir is not None:
            if len(bright_pos) >= 5 and save_image:
                count += 1
                print(f'count:{count} num_points:{len(bright_pos)} theta:{theta} rate:{count/(time.time()-st)*60}/min')
                cv2.imwrite(f'{output_dir}/count{count}_raw.png', raw)
                cv2.imwrite(f'{output_dir}/count{count}_masked.png', masked)
                cv2.imwrite(f'{output_dir}/count{count}_lined.png', lined)
                cv2.imwrite(f'{output_dir}/count{count}_points.png', points)
                cv2.imwrite(f'{output_dir}/count{count}_detect.png', detect)

    cv2.imshow('frame', frame)
    if movie is not None:
        movie.write(frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        param_dict = {}
        for h, name in zip(line_height, line_names):
            param_dict[name] = h
        param_dict['threshold'] = threshold
        param_dict['exposure'] = cv2.getTrackbarPos('exposure', 'controller')
        param_dict['focus'] = cv2.getTrackbarPos('focus', 'controller')
        with open('param.json', 'w') as f:
            json.dump(param_dict, f, indent=4)


if output_dir is not None:
    if movie is not None:
        movie.release()

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap.release()
cv2.destroyAllWindows()
