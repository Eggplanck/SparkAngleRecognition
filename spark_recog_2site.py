import numpy as np
import cv2
import json
import os

output_file = 'output_2site.csv'


class Site():
    def __init__(self, cap, window_name, param_file):
        self.cap = cap
        self.window_name = window_name
        ret, self.frame = self.cap.read()
        assert ret

        # cap.set(cv2.CAP_PROP_SETTINGS, 1)
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 100)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = width
        self.height = height

        cv2.namedWindow(window_name)

        def nothing(x):
            pass

        line_num = 8
        self.line_names = []
        for l in range(0, line_num):
            line_name = str(l)
            cv2.createTrackbar(line_name, window_name,
                               (height//line_num)*l, height, nothing)
            self.line_names.append(line_name)

        # threshold等の値はデバイスによって変わる
        cv2.createTrackbar('threshold', window_name, 150, 255, nothing)

        def set_exposure(value):
            re = self.cap.set(cv2.CAP_PROP_EXPOSURE, -10+value)
            print(re)

        cv2.createTrackbar('exposure', window_name, 5, 8, set_exposure)

        def set_focus(value):
            re = self.cap.set(cv2.CAP_PROP_FOCUS, 100+value*5)
            print(re)

        cv2.createTrackbar('focus', window_name, 0, 120, set_focus)

        self.param_file = param_file
        if os.path.exists(param_file):
            with open(param_file) as f:
                param_dict = json.load(f)
            for key, value in param_dict.items():
                cv2.setTrackbarPos(key, window_name, value)

    def make_lines(self, frame):
        # パラメータの値を取得
        line_height = [cv2.getTrackbarPos(line_name, self.window_name)
                       for line_name in self.line_names]

        # 区切り線の描画
        for h, name in zip(line_height, self.line_names):
            frame = cv2.line(frame, (0, h), (self.width, h), (255, 0, 0), 5)
            frame = cv2.putText(
                frame, name, (0, h+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    def bright_pos(self, frame):
        line_height = [cv2.getTrackbarPos(line_name, self.window_name)
                       for line_name in self.line_names]
        threshold = cv2.getTrackbarPos('threshold', self.window_name)
        # 明るい点のマスクを作成
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = gray > threshold

        frame[mask] = [0, 255, 0]

        # 各区間で明るい点の重心を計算
        bright_pos = []
        for l in range(len(line_height)-1):
            upper = line_height[l]
            lower = line_height[l+1]
            bright = mask[upper:lower, :]
            if bright.any() and lower-upper > 0:
                x_pos = np.repeat(np.arange(0, self.width)[
                    None, :], lower-upper, axis=0)
                bright_cm = x_pos[bright].sum()/bright.sum()
                bright_pos.append((bright_cm, (upper+lower)/2))

        return frame, bright_pos

    def linear_reg(self, frame, bright_pos):
        if len(bright_pos) < 2:
            return frame, None

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
        elif self.width <= b:
            p0 = (self.width-1, int((self.width-1-b)/a))
        else:
            p0 = (int(b), 0)

        x1 = a * self.height + b
        if x1 < 0:
            p1 = (0, int(-b/a))
        elif self.width <= x1:
            p1 = (self.width-1, int((self.width-1-b)/a))
        else:
            p1 = (int(x1), self.height-1)
        frame = cv2.line(frame, p0, p1, (0, 255, 255), 5)
        frame = cv2.putText(frame, '{:.2f} degrees'.format(np.degrees(
            theta)), (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return frame, {'theta': theta, 'res': res, 'points': len(bright_pos)}

    def process(self):
        ret, frame = self.cap.read()
        assert ret

        frame, bright_pos = self.bright_pos(frame)
        frame = self.make_lines(frame)
        frame, data = self.linear_reg(frame, bright_pos)

        for posx, posy in bright_pos:
            frame = cv2.circle(
                frame, (int(posx), int(posy)), 5, (0, 0, 255), -1)

        cv2.imshow(self.window_name, frame)
        return data

    def save_param(self):
        line_height = [cv2.getTrackbarPos(line_name, self.window_name)
                       for line_name in self.line_names]
        param_dict = {}
        for h, name in zip(line_height, self.line_names):
            param_dict[name] = h
        param_dict['threshold'] = cv2.getTrackbarPos(
            'threshold', self.window_name)
        param_dict['exposure'] = cv2.getTrackbarPos(
            'exposure', self.window_name)
        param_dict['focus'] = cv2.getTrackbarPos('focus', self.window_name)
        with open(self.param_file, 'w') as f:
            json.dump(param_dict, f, indent=4)


cap1 = cv2.VideoCapture(2)
site1 = Site(cap1, window_name='frame1', param_file='param1.json')
cap2 = cv2.VideoCapture(0)
site2 = Site(cap2, window_name='frame2', param_file='param2.json')


if output_file is not None:
    with open(output_file, 'w') as f:
        f.write('theta1,residue1,points1,theta2,residue2,points2')
    file = None

    def change_save_mode(value):
        global file
        if value == 1 and file is None:
            file = open(output_file, 'a')
        elif value == 0 and file is not None:
            file.close()
            file = None
    cv2.namedWindow('save')
    cv2.createTrackbar('save mode', 'save', 0, 1, change_save_mode)


while site1.cap.isOpened() and site2.cap.isOpened():
    data1 = site1.process()
    data2 = site2.process()
    if data1 is not None and data2 is not None:
        print(f"theta1:{data1['theta']}, theta2:{data2['theta']}")
        if file is not None:
            file.write(
                f"\n{data1['theta']},{data1['res']},{data1['points']},{data2['theta']},{data2['res']},{data2['points']}")

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        site1.save_param()
        site2.save_param()


if output_file is not None:
    if file is not None:
        file.close()

site1.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
site2.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
site1.cap.release()
site2.cap.release()
cv2.destroyAllWindows()
