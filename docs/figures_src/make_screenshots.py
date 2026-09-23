"""
Capture real screenshots of the ClassRoomSTORM application windows for the
Help document.

The native Windows Qt platform is used so system fonts render correctly (the
'offscreen' plugin renders text as missing-glyph boxes). Each window is built
and grabbed with QWidget.grab(); windows are kept off the desktop while their native widgets are rendered. Because every
workflow app creates its own QApplication, each window is captured in a
separate subprocess.

  python make_screenshots.py            -> capture all + build montage
  python make_screenshots.py launcher
  python make_screenshots.py reconstruction
  python make_screenshots.py virtual
  python make_screenshots.py results

Output: ../figures/help/
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
APPS = os.path.join(REPO, "apps")
HELP = os.path.abspath(os.path.join(HERE, "..", "figures", "help"))
for p in (REPO, APPS):
    if p not in sys.path:
        sys.path.insert(0, p)


def _pump(app, n=16):
    for _ in range(n):
        app.processEvents()


def capture_launcher():
    """H2: the Studio launcher (module-level StudioWindow)."""
    from PySide6 import QtWidgets
    import ClassRoomSTORM_Studio as studio
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = studio.StudioWindow()
    win.resize(1200, 730)
    from PySide6.QtCore import Qt
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    win.show()
    _pump(app)
    try:
        win.background.skip_intro()
    except Exception:
        pass
    _pump(app)
    win.grab().save(os.path.join(HELP, "H2_launcher_screenshot.png"))
    print("OK H2_launcher_screenshot.png")


def capture_via_run_gui(module_name, out_name, nav_row=None):
    """Capture a workflow window whose class is nested inside run_gui().

    nav_row : if set, select that row of the window's main QListWidget before
              grabbing (used to reach the results page).
    """
    from PySide6 import QtWidgets
    import importlib
    mod = importlib.import_module(module_name)
    state = {"saved": False}

    def fake_exec(app_self, *args, **kwargs):
        app = QtWidgets.QApplication.instance()
        _pump(app, 18)
        tops = [w for w in app.topLevelWidgets() if w.isWindow()]
        if tops:
            win = max(tops, key=lambda w: w.width() * w.height())
            if nav_row is not None:
                nav = win.findChild(QtWidgets.QListWidget)
                if nav is not None and nav.count() > nav_row:
                    nav.setCurrentRow(nav_row)
                    _pump(app, 14)
            win.grab().save(os.path.join(HELP, out_name))
            state["saved"] = True
        return 0

    QtWidgets.QApplication.exec = fake_exec
    QtWidgets.QApplication.exec_ = fake_exec
    from PySide6.QtCore import Qt
    native_show = QtWidgets.QWidget.show
    def hidden_show(widget):
        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        return native_show(widget)
    QtWidgets.QWidget.show = hidden_show
    mod.run_gui()
    print(("OK " if state["saved"] else "FAILED ") + out_name)


def build_montage():
    """H3: a readable two-window walkthrough (Virtual Experiment, then SR
    Reconstruction) stacked vertically so the GUI text stays legible."""
    from PIL import Image, ImageOps
    parts = ["H3b_virtual_experiment.png", "H6_reconstruction_load_crop.png"]
    imgs = []
    for fname in parts:
        path = os.path.join(HELP, fname)
        if os.path.exists(path):
            im = Image.open(path).convert("RGB")
            im = ImageOps.contain(im, (1700, 1700))
            im = ImageOps.expand(im, border=1, fill=(165, 165, 165))
            imgs.append(im)
    if not imgs:
        print("FAILED H3_quickstart_montage.png (no source screenshots)")
        return
    gap = 26
    w = max(im.width for im in imgs)
    h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    montage = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for im in imgs:
        montage.paste(im, ((w - im.width) // 2, y))
        y += im.height + gap
    montage.save(os.path.join(HELP, "H3_quickstart_montage.png"))
    print("OK H3_quickstart_montage.png")


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "launcher":
            capture_launcher()
        elif target == "reconstruction":
            capture_via_run_gui("ClassRoomSTORM_Reconstruction",
                                "H6_reconstruction_load_crop.png")
        elif target == "virtual":
            capture_via_run_gui("ClassRoomSTORM_VirtualExperiment",
                                "H3b_virtual_experiment.png")
        elif target == "results":
            capture_via_run_gui("ClassRoomSTORM_Reconstruction",
                                "H7_results_browser.png", nav_row=6)
        return

    os.makedirs(HELP, exist_ok=True)
    for target in ("launcher", "reconstruction", "virtual", "results"):
        try:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), target],
                               capture_output=True, text=True, timeout=200)
            tail = (r.stdout or "").strip().splitlines()
            print("[%s] exit=%d %s" % (target, r.returncode,
                                       tail[-1] if tail else ""))
            if r.returncode != 0 and r.stderr:
                print("   " + r.stderr.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            print("[%s] TIMEOUT" % target)
    build_montage()


if __name__ == "__main__":
    main()
