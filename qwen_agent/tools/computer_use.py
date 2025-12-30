import os
import time
import uuid
from typing import Union, Tuple, List

import pyautogui

from qwen_agent.settings import DEFAULT_WORKSPACE

from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("computer_use")
class ComputerUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
* `screenshot`: Capture a screenshot without performing any other action.
""".strip(),
                "enum": [
                    "key",
                    "type",
                    "mouse_move",
                    "left_click",
                    "left_click_drag",
                    "right_click",
                    "middle_click",
                    "double_click",
                    "triple_click",
                    "scroll",
                    "hscroll",
                    "wait",
                    "terminate",
                    "answer",
                    "screenshot",
                ],
                "type": "string",
            },
            "keys": {
                "description": "Required only by `action=key`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=type` and `action=answer`.",
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to.",
                "type": "array",
            },
            "pixels": {
                "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll` and `action=hscroll`.",
                "type": "number",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=wait`.",
                "type": "number",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        cfg = cfg or {}
        if "display_width_px" in cfg and "display_height_px" in cfg:
            self.display_width_px = cfg["display_width_px"]
            self.display_height_px = cfg["display_height_px"]
        else:
            size = pyautogui.size()
            self.display_width_px = size.width
            self.display_height_px = size.height
        self.work_dir = cfg.get("work_dir", os.path.join(DEFAULT_WORKSPACE, "tools", "computer_use"))
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action in ["left_click", "right_click", "middle_click", "double_click", "triple_click"]:
            return self._mouse_click(action, params["coordinate"])
        elif action == "key":
            return self._key(params["keys"])
        elif action == "type":
            return self._type(params["text"])
        elif action == "mouse_move":
            return self._mouse_move(params["coordinate"])
        elif action == "left_click_drag":
            return self._left_click_drag(params["coordinate"])
        elif action == "scroll":
            return self._scroll(params["pixels"])
        elif action == "hscroll":
            return self._hscroll(params["pixels"])
        elif action == "answer":
            return self._answer(params["text"])
        elif action == "screenshot":
            return self._screenshot_result(action="screenshot")
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Invalid action: {action}")

    def _mouse_click(self, button: str, coordinate: Tuple[int, int]):
        x, y = self._coerce_coord(coordinate)
        clicks = 1
        if button == "double_click":
            clicks = 2
            button = "left"
        elif button == "triple_click":
            clicks = 3
            button = "left"
        elif button == "left_click":
            button = "left"
        elif button == "right_click":
            button = "right"
        elif button == "middle_click":
            button = "middle"
        pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=0.1)
        return self._screenshot_result(action="click")

    def _key(self, keys: List[str]):
        if not keys:
            raise ValueError("keys cannot be empty for action=key")
        for key in keys:
            pyautogui.keyDown(key)
        for key in reversed(keys):
            pyautogui.keyUp(key)
        return self._screenshot_result(action="key")

    def _type(self, text: str):
        pyautogui.typewrite(text)
        return self._screenshot_result(action="type")

    def _mouse_move(self, coordinate: Tuple[int, int]):
        x, y = self._coerce_coord(coordinate)
        pyautogui.moveTo(x, y)
        return self._screenshot_result(action="mouse_move")

    def _left_click_drag(self, coordinate: Tuple[int, int]):
        x, y = self._coerce_coord(coordinate)
        pyautogui.dragTo(x, y, button="left")
        return self._screenshot_result(action="left_click_drag")

    def _scroll(self, pixels: int):
        pyautogui.scroll(int(pixels))
        return self._screenshot_result(action="scroll")

    def _hscroll(self, pixels: int):
        if hasattr(pyautogui, "hscroll"):
            pyautogui.hscroll(int(pixels))
        else:
            pyautogui.scroll(int(pixels))
        return self._screenshot_result(action="hscroll")

    def _answer(self, text: str):
        return {"answer": text}

    def _screenshot(self):
        return self._screenshot_result(action="screenshot")

    def _wait(self, time: int):
        time = float(time)
        time = max(0.0, time)
        time.sleep(time)
        return self._screenshot_result(action="wait")

    def _terminate(self, status: str):
        if status not in ("success", "failure"):
            raise ValueError("status must be 'success' or 'failure'")
        return {"status": status}

    @staticmethod
    def _coerce_coord(coord: Tuple[int, int]) -> Tuple[int, int]:
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            raise ValueError("coordinate must be [x, y]")
        return int(coord[0]), int(coord[1])

    def _screenshot_result(self, action: str) -> dict:
        os.makedirs(self.work_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(self.work_dir, filename)
        image = pyautogui.screenshot()
        image.save(path)
        return {"action": action, "screenshot": path}
