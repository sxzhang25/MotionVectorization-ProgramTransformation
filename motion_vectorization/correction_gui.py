import sys
import os
import pickle
import numpy as np
import cv2
from PIL import Image as Img
from PIL import ImageTk
from tkinter import *

from .utils import get_numbers, get_shape_centroid, get_alpha, get_shape_coords

### HELPER ###
def display_frame():
    global G_COLOR_MAP
    global G_CURR_FRAME_DISPLAY
    if G_MASK_ON:
        track_vis = G_CURR_FRAME_RGB.copy()
        t_info = G_TIME_BANK['shapes'][G_CURR_FRAME_IDX]
        for label in t_info:
            if (label + 1) not in G_COLOR_MAP:
                G_COLOR_MAP[label + 1] = G_RNG.integers(low=128, high=255, size=(3,))
            track_vis[t_info[label]['mask'][50:-50, 50:-50]>=0, :] = G_COLOR_MAP[label + 1]
        for label in t_info:
            cx, cy = get_shape_centroid(np.uint8(t_info[label]['mask']>=0)[50:-50, 50:-50])
            track_vis = cv2.circle(track_vis, (int(cx), int(cy)), 1, (255, 255, 255), 2)
            track_vis = cv2.putText(
                track_vis, f'{str(label)}', (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        track_vis = Img.fromarray(cv2.cvtColor(track_vis, cv2.COLOR_RGB2BGR))
    else:
        track_vis = Img.fromarray(cv2.cvtColor(G_CURR_FRAME_RGB, cv2.COLOR_RGB2BGR))
    frame_height, frame_width = track_vis.size
    resize_ratio = min(1.0, G_MAX_FRAME_HEIGHT / frame_height)
    track_vis = track_vis.resize((G_MAX_FRAME_HEIGHT, int(resize_ratio * frame_width)), Img.BICUBIC)
    G_CURR_FRAME_DISPLAY = ImageTk.PhotoImage(track_vis)
    Label(
        G_DISPLAY_WIN, 
        image=G_CURR_FRAME_DISPLAY, 
        bg='grey'
    ).grid(row=0, column=0, padx=5, pady=5)
    G_FRAME_ENTRY.delete(0, END)
    G_FRAME_ENTRY.insert(0, str(G_CURR_FRAME_IDX))
    # print(G_TIME_BANK['shapes'][G_CURR_FRAME_IDX].keys())


### BUTTON FUNCTIONS ###
def clicked():
    '''if button is clicked, display message'''
    print('Clicked.')


def next_frame():
    global G_CURR_FRAME_IDX
    global G_CURR_FRAME_RGB
    if G_CURR_FRAME_IDX == G_MAX_FRAME - 1:
        return
    else:
        G_CURR_FRAME_IDX = G_CURR_FRAME_IDX + 1
        t = G_FRAME_IDXS[G_CURR_FRAME_IDX]
        G_CURR_FRAME_RGB = cv2.imread(os.path.join(G_RGB_FOLDER, f'{t:03d}.png'))
        display_frame()


def prev_frame():
    global G_CURR_FRAME_IDX
    global G_CURR_FRAME_RGB
    if G_CURR_FRAME_IDX == 0:
        return
    else:
        G_CURR_FRAME_IDX = G_CURR_FRAME_IDX - 1
        t = G_FRAME_IDXS[G_CURR_FRAME_IDX]
        G_CURR_FRAME_RGB = cv2.imread(os.path.join(G_RGB_FOLDER, f'{t:03d}.png'))
        display_frame()


def replace_frame():
    global G_TIME_BANK
    old_idx = int(G_SELECT_ENTRY.get())
    new_idx = int(G_REPLACE_ENTRY.get())
    if old_idx in G_TIME_BANK['shapes'][G_CURR_FRAME_IDX]:
        if new_idx in G_TIME_BANK['shapes'][G_CURR_FRAME_IDX]:
            old_idx_mask = G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][old_idx]['mask']
            G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][new_idx]['mask'][old_idx_mask>=0] = new_idx
        else:
            G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][new_idx] = G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][old_idx]
            new_mask = G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][new_idx]['mask'].copy()
            new_mask[new_mask>=0] = new_idx
            G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][new_idx]['mask'] = new_mask
        del G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][old_idx]
        min_x, min_y, max_x, max_y = get_shape_coords(G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][new_idx]['mask'])
        G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][new_idx]['coords'] = ((min_x, min_y), (max_x, max_y))
    display_frame()


def replace_after_frames():
    global G_TIME_BANK
    old_idx = int(G_SELECT_ENTRY.get())
    new_idx = int(G_REPLACE_ENTRY.get())
    for t in range(G_CURR_FRAME_IDX, G_MAX_FRAME):
        if old_idx in G_TIME_BANK['shapes'][t]:
            if new_idx in G_TIME_BANK['shapes'][t]:
                old_idx_mask = G_TIME_BANK['shapes'][t][old_idx]['mask']
                G_TIME_BANK['shapes'][t][new_idx]['mask'][old_idx_mask>=0] = new_idx
            else:
                G_TIME_BANK['shapes'][t][new_idx] = G_TIME_BANK['shapes'][t][old_idx]
                new_mask = G_TIME_BANK['shapes'][t][new_idx]['mask'].copy()
                new_mask[new_mask>=0] = new_idx
                G_TIME_BANK['shapes'][t][new_idx]['mask'] = new_mask
            del G_TIME_BANK['shapes'][t][old_idx]
            min_x, min_y, max_x, max_y = get_shape_coords(G_TIME_BANK['shapes'][t][new_idx]['mask'])
            G_TIME_BANK['shapes'][t][new_idx]['coords'] = ((min_x, min_y), (max_x, max_y))
    display_frame()


def toggle_mask():
    global G_MASK_ON
    G_MASK_ON = (not G_MASK_ON)
    display_frame()


def jump_to():
    global G_CURR_FRAME_IDX
    global G_CURR_FRAME_RGB
    frame_idx = int(G_FRAME_ENTRY.get())
    if frame_idx >= 0 and frame_idx < G_MAX_FRAME:
        G_CURR_FRAME_IDX = frame_idx
        t = G_FRAME_IDXS[G_CURR_FRAME_IDX]
        G_CURR_FRAME_RGB = cv2.imread(os.path.join(G_RGB_FOLDER, f'{t:03d}.png'))
    display_frame()


def delete():
    global G_TIME_BANK
    select_idx = int(G_SELECT_ENTRY.get())
    for t in range(G_CURR_FRAME_IDX, G_MAX_FRAME):
        if select_idx in G_TIME_BANK['shapes'][t]:
            del G_TIME_BANK['shapes'][t][select_idx]
    display_frame()


def copy():
    global G_COPY_INFO
    select_idx = int(G_SELECT_ENTRY.get())
    if select_idx in G_TIME_BANK['shapes'][G_CURR_FRAME_IDX]:
        G_COPY_INFO = G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]


def paste():
    global G_TIME_BANK
    if G_COPY_INFO is None:
        return
    select_idx = int(G_SELECT_ENTRY.get())
    if select_idx in G_TIME_BANK['shapes'][G_CURR_FRAME_IDX]:
        G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['mask'][G_COPY_INFO['mask']==select_idx] = select_idx
        # Update bounding box coordinates.
        G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['coords'][0][0] = \
            min(G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['coords'][0][0], G_COPY_INFO['coords'][0][0])
        G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['coords'][0][1] = \
            min(G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['coords'][0][1], G_COPY_INFO['coords'][0][1])
        G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['coords'][1][0] = \
            max(G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['coords'][1][0], G_COPY_INFO['coords'][1][0])
        G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['coords'][1][1] = \
            max(G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['coords'][1][1], G_COPY_INFO['coords'][1][1])
    else:
        G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx] = G_COPY_INFO
    display_frame()


def update_canonical_img():
    select_idx = int(G_SELECT_ENTRY.get())
    if select_idx in G_TIME_BANK['shapes'][G_CURR_FRAME_IDX]:
        old_alpha = np.float64(G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['mask'][50:-50, 50:-50]>=0)
        new_alpha = get_alpha(
            old_alpha, 
            G_CURR_FRAME_RGB,
            kernel_radius=11,
            expand=True
        )
        min_x, min_y, max_x, max_y = get_shape_coords(new_alpha)
        new_shape_rgba = np.concatenate([G_CURR_FRAME_RGB, np.uint8(255 * new_alpha)], axis=-1)
        new_shape_rgba = new_shape_rgba[min_y:max_y, min_x:max_x]
        G_TIME_BANK['shapes'][G_CURR_FRAME_IDX][select_idx]['coords'] = ((min_x, min_y), (max_x, max_y))
        cv2.imwrite(os.path.join('outputs', f'{G_VIDEO_NAME}_{G_SUFFIX}', 'shapes', f'{select_idx}.png'), new_shape_rgba)


def save():
    outfile = os.path.join('outputs', f'{G_VIDEO_NAME}_{G_SUFFIX}', 'time_bank.pkl')
    with open(outfile, 'wb') as handle:
        pickle.dump(G_TIME_BANK, handle)
    print('Saved to:', outfile)


def main():
    global G_SEED
    global G_RNG
    global G_MASK_ON
    global G_CURR_FRAME_RGB
    global G_MAX_FRAME_HEIGHT
    global G_DISPLAY_WIN
    global G_CURR_FRAME_IDX
    global G_MAX_FRAME
    global G_TIME_BANK
    global G_FRAME_IDXS
    global G_COLOR_MAP
    global G_CURR_FRAME_DISPLAY
    global G_REPLACE_ENTRY
    global G_SELECT_ENTRY
    global G_TIME_BANK_FILE
    global G_RGB_FOLDER
    global G_VIDEO_NAME
    global G_SUFFIX
    global G_FRAME_ENTRY
    global G_COPY_INFO
    global G_FRAME_WIDTH
    global G_FRAME_HEIGHT

    G_MASK_ON = True
    G_MAX_FRAME_HEIGHT = 450
    G_CURR_FRAME_IDX = 0
    G_SEED = 0
    G_RNG = np.random.default_rng(seed=G_SEED)
    G_COLOR_MAP = {0: [0, 0, 0]}

    root = Tk()  # create root window
    root.title('Motion graphics tracking editor')
    root.maxsize(900, 900)  # width x height
    root.config(bg='skyblue')

    G_VIDEO_NAME = sys.argv[1]
    G_SUFFIX = 'None'

    G_RGB_FOLDER = os.path.join('videos', G_VIDEO_NAME, 'rgb')
    G_FRAME_IDXS = get_numbers(G_RGB_FOLDER)
    G_TIME_BANK_FILE = os.path.join('outputs', f'{G_VIDEO_NAME}_{G_SUFFIX}', 'time_bank.pkl')
    G_TIME_BANK = pickle.load(open(G_TIME_BANK_FILE, 'rb'))
    G_MAX_FRAME = len(G_TIME_BANK['shapes'])

    G_CURR_FRAME_RGB = cv2.imread(os.path.join(G_RGB_FOLDER, '001.png'))
    G_FRAME_HEIGHT, G_FRAME_WIDTH, _ = G_CURR_FRAME_RGB.shape
    G_COPY_INFO = None

    # Create left and right frames
    left_frame = Frame(root, width=200, height=400, bg='grey')
    left_frame.grid(row=0, column=0, padx=10, pady=5)

    G_DISPLAY_WIN = Frame(root, width=650, height=500, bg='grey')
    G_DISPLAY_WIN.grid(row=0, column=1, padx=10, pady=5)

    scrub_frame = Frame(root)
    scrub_frame.grid(row=2, column=1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    backward_btn = Button(
        scrub_frame,
        text='< Prev',
        command=prev_frame
    ).grid(row=0, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    forward_btn = Button(
        scrub_frame,
        text='Next >',
        command=next_frame
    ).grid(row=0, column=1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    G_FRAME_ENTRY = Entry(scrub_frame)
    G_FRAME_ENTRY.insert(0, str(G_CURR_FRAME_IDX))
    G_FRAME_ENTRY.grid(row=0, column=2, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    Label(
        scrub_frame,
        text=f'/{G_MAX_FRAME}'
    ).grid(row=0, column=3, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    jump_btn = Button(
        scrub_frame,
        text='Jump to frame',
        command=jump_to
    ).grid(row=0, column=4, padx=5, pady=5, sticky='w'+'e'+'n'+'s')

    tool_bar = Frame(
        left_frame, 
        width=180, 
        height=185, 
        bg='grey'
    )
    tool_bar.grid(row=2, column=0, padx=5, pady=5)

    # For now, when the buttons are clicked, they only call the clicked() method. We will add functionality later.
    toggle_mask_btn = Button(
        tool_bar,
        text='Toggle masks',
        command=toggle_mask
    ).grid(row=1, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    id_frame = Frame(tool_bar)
    id_frame.grid(row=2, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    Label(
        id_frame,
        text='Select ID'
    ).grid(row=0, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    G_SELECT_ENTRY = Entry(id_frame)
    G_SELECT_ENTRY.grid(row=1, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    Label(
        id_frame,
        text='Replace ID'
    ).grid(row=0, column=1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    G_REPLACE_ENTRY = Entry(id_frame)
    G_REPLACE_ENTRY.grid(row=1, column=1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    replace_curr_btn = Button(
        tool_bar,
        text='Replace on this frame',
        command=replace_frame
    ).grid(row=4, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    replace_after_btn = Button(
        tool_bar,
        text='Replace on all subsequent frames',
        command=replace_after_frames
    ).grid(row=5, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    delete_btn = Button(
        tool_bar,
        text='Delete selected ID',
        command=delete
    ).grid(row=6, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    update_canon_btn = Button(
        tool_bar,
        text='Update selected ID image',
        command=update_canonical_img
    ).grid(row=7, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    copy_paste_frame = Frame(tool_bar)
    copy_paste_frame.grid(row=8, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    copy_btn = Button(
        copy_paste_frame,
        text='Copy selected ID',
        command=copy
    ).grid(row=0, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    paste_btn = Button(
        copy_paste_frame,
        text='Paste selected ID',
        command=paste
    ).grid(row=0, column=1, padx=5, pady=5, sticky='w'+'e'+'n'+'s')
    save_btn = Button(
        tool_bar,
        text='Save',
        command=save
    ).grid(row=9, column=0, padx=5, pady=5, sticky='w'+'e'+'n'+'s')

    display_frame()

    root.mainloop()


if __name__ == '__main__':
    main()