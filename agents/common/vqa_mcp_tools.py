import json
import mimetypes
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency in some environments
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _guess_mime_type(path: str) -> str:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "image/png"


def _sanitize_filename(name: str, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def _ensure_pillow() -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for visual tools")


class VQAImageSession:
    def __init__(self, task_id: str):
        self.task_id = str(task_id)
        self._temp_dir = tempfile.TemporaryDirectory(prefix=f"vqa_tools_{self.task_id}_")
        self._image_dir = Path(self._temp_dir.name)
        self._images: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    @property
    def workdir(self) -> str:
        return self._temp_dir.name

    def destroy(self) -> None:
        self._temp_dir.cleanup()

    def register_initial_images(self, image_refs: Iterable[str]) -> None:
        for image_ref in image_refs:
            self._register_reference(image_ref)

    def list_images(self) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []
        for image_id, record in sorted(self._images.items()):
            hydrated = dict(record)
            try:
                hydrated = dict(self._ensure_image_ready(image_id))
            except Exception:
                pass
            images.append(hydrated)
        return images

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        return dict(self._ensure_image_ready(image_id))

    def crop_image(
        self,
        image_id: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> Dict[str, Any]:
        _ensure_pillow()
        record = self._ensure_image_ready(image_id)
        with Image.open(record["path"]) as image:
            width, height = image.size
            left = max(0, int(left))
            top = max(0, int(top))
            right = min(width, int(right))
            bottom = min(height, int(bottom))
            if left >= right or top >= bottom:
                raise ValueError("crop coordinates must define a non-empty region")
            cropped = image.crop((left, top, right, bottom))
            new_record = self._save_generated_image(
                cropped,
                source_type="crop",
                source_image_id=image_id,
            )

        return {
            "image_id": new_record["image_id"],
            "path": new_record["path"],
            "width": new_record["width"],
            "height": new_record["height"],
            "crop_box": [left, top, right, bottom],
            "source_image_id": image_id,
        }

    def zoom_image(self, image_id: str, factor: float) -> Dict[str, Any]:
        _ensure_pillow()
        record = self._ensure_image_ready(image_id)
        zoom_factor = float(factor)
        if zoom_factor <= 1.0:
            raise ValueError("zoom factor must be greater than 1.0")

        with Image.open(record["path"]) as image:
            width, height = image.size
            crop_width = max(1, int(round(width / zoom_factor)))
            crop_height = max(1, int(round(height / zoom_factor)))
            left = max(0, (width - crop_width) // 2)
            top = max(0, (height - crop_height) // 2)
            cropped = image.crop((left, top, left + crop_width, top + crop_height))
            resampled = cropped.resize((width, height), Image.Resampling.LANCZOS)
            new_record = self._save_generated_image(
                resampled,
                source_type="zoom",
                source_image_id=image_id,
            )

        return {
            "image_id": new_record["image_id"],
            "path": new_record["path"],
            "width": new_record["width"],
            "height": new_record["height"],
            "factor": zoom_factor,
            "source_image_id": image_id,
        }

    def resize_image(
        self,
        image_id: str,
        target_width: int,
        target_height: int,
    ) -> Dict[str, Any]:
        _ensure_pillow()
        record = self._ensure_image_ready(image_id)
        width = int(target_width)
        height = int(target_height)
        if width <= 0 or height <= 0:
            raise ValueError("target dimensions must be positive integers")

        with Image.open(record["path"]) as image:
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
            new_record = self._save_generated_image(
                resized,
                source_type="resize",
                source_image_id=image_id,
            )

        return {
            "image_id": new_record["image_id"],
            "path": new_record["path"],
            "width": new_record["width"],
            "height": new_record["height"],
            "source_image_id": image_id,
        }

    def _materialize_image_ref(self, image_ref: str) -> Tuple[str, str]:
        if image_ref.startswith(("http://", "https://")):
            parsed = urlparse(image_ref)
            file_name = _sanitize_filename(Path(parsed.path).name, f"remote_{self._counter}.img")
            output_path = self._image_dir / file_name
            urllib.request.urlretrieve(image_ref, output_path)
            return str(output_path), "remote"

        return os.path.abspath(image_ref), "original"

    def _register_reference(self, image_ref: str) -> Dict[str, Any]:
        image_id = f"image_{self._counter}"
        self._counter += 1
        record: Dict[str, Any] = {
            "image_id": image_id,
            "source_type": "remote" if image_ref.startswith(("http://", "https://")) else "original",
            "source_ref": image_ref,
        }
        if not image_ref.startswith(("http://", "https://")):
            record["path"] = os.path.abspath(image_ref)
            record.update(self._read_image_metadata(record["path"]))
            record["mime_type"] = _guess_mime_type(record["path"])
        self._images[image_id] = record
        return record

    def _register_image(
        self,
        image_path: str,
        *,
        source_type: str,
        source_image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        _ensure_pillow()
        image_id = f"image_{self._counter}"
        self._counter += 1
        with Image.open(image_path) as image:
            width, height = image.size
        record = {
            "image_id": image_id,
            "path": os.path.abspath(image_path),
            "width": width,
            "height": height,
            "mime_type": _guess_mime_type(image_path),
            "source_type": source_type,
        }
        if source_image_id:
            record["source_image_id"] = source_image_id
        self._images[image_id] = record
        return record

    def _read_image_metadata(self, image_path: str) -> Dict[str, Any]:
        _ensure_pillow()
        with Image.open(image_path) as image:
            width, height = image.size
        return {"width": width, "height": height}

    def _save_generated_image(
        self,
        image: "Image.Image",
        *,
        source_type: str,
        source_image_id: str,
    ) -> Dict[str, Any]:
        output_path = self._image_dir / f"{source_type}_{self._counter}.png"
        image.save(output_path)
        return self._register_image(
            str(output_path),
            source_type=source_type,
            source_image_id=source_image_id,
        )

    def _get_image_record(self, image_id: str) -> Dict[str, Any]:
        normalized_id = str(image_id)
        if normalized_id in self._images:
            return self._images[normalized_id]

        if normalized_id.lower() == "original":
            candidate_id = "image_0"
            if candidate_id in self._images:
                return self._images[candidate_id]

        # Be lenient with common model outputs like "0" instead of "image_0".
        if normalized_id.isdigit():
            candidate_id = f"image_{normalized_id}"
            if candidate_id in self._images:
                return self._images[candidate_id]

        raise ValueError(f"Unknown image_id: {image_id}")

    def _ensure_image_ready(self, image_id: str) -> Dict[str, Any]:
        record = self._get_image_record(image_id)
        if "path" in record and "width" in record and "height" in record:
            return record

        local_path, source_type = self._materialize_image_ref(str(record["source_ref"]))
        record["path"] = os.path.abspath(local_path)
        record["source_type"] = source_type
        record["mime_type"] = _guess_mime_type(record["path"])
        record.update(self._read_image_metadata(record["path"]))
        return record


class VQAToolRegistry:
    def __init__(self, python_executor: Any, image_session: VQAImageSession):
        self.python_executor = python_executor
        self.image_session = image_session
        self._tools = {
            "execute_python": self._execute_python,
            "list_images": self._list_images,
            "get_image_info": self._get_image_info,
            "crop_image": self._crop_image,
            "zoom_image": self._zoom_image,
            "resize_image": self._resize_image,
        }

    @staticmethod
    def get_tool_schemas() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": (
                        "Execute Python code in a persistent interpreter. "
                        "Use this to verify calculations, count objects, manipulate numbers, "
                        "solve equations, or check candidate answers."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute. Print any values you want to inspect.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_images",
                    "description": "List images available for the current task, including derived images created by tools.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_image_info",
                    "description": "Return width, height, format, and path metadata for an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_id": {"type": "string", "description": "The image id returned by list_images."}
                        },
                        "required": ["image_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "crop_image",
                    "description": "Crop an image using pixel coordinates and create a new derived image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_id": {"type": "string"},
                            "left": {"type": "integer"},
                            "top": {"type": "integer"},
                            "right": {"type": "integer"},
                            "bottom": {"type": "integer"},
                        },
                        "required": ["image_id", "left", "top", "right", "bottom"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "zoom_image",
                    "description": "Zoom into the center of an image by a factor greater than 1 and create a new derived image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_id": {"type": "string"},
                            "factor": {"type": "number"},
                        },
                        "required": ["image_id", "factor"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "resize_image",
                    "description": "Resize an image to the requested width and height and create a new derived image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_id": {"type": "string"},
                            "target_width": {"type": "integer"},
                            "target_height": {"type": "integer"},
                        },
                        "required": ["image_id", "target_width", "target_height"],
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._tools.get(tool_name)
        if handler is None:
            return self._error_result("UnsupportedTool", f"Unsupported tool: {tool_name}")

        try:
            return handler(arguments)
        except Exception as exc:
            return self._error_result(type(exc).__name__, str(exc))

    def _execute_python(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        code = str(arguments.get("code", ""))
        execution_result = self.python_executor.execute_code(code)
        output = str(execution_result.get("output", "(No output)"))
        return {
            "output": output,
            "error_type": execution_result.get("error_type"),
            "tool_payload": {"ok": execution_result.get("error_type") is None, "code": code},
        }

    def _list_images(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        del arguments
        payload = {"ok": True, "images": self.image_session.list_images()}
        return {"output": json.dumps(payload, ensure_ascii=False), "error_type": None, "tool_payload": payload}

    def _get_image_info(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"ok": True, **self.image_session.get_image_info(str(arguments.get("image_id", "")))}
        return {"output": json.dumps(payload, ensure_ascii=False), "error_type": None, "tool_payload": payload}

    def _crop_image(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "ok": True,
            **self.image_session.crop_image(
                str(arguments.get("image_id", "")),
                int(arguments.get("left", 0)),
                int(arguments.get("top", 0)),
                int(arguments.get("right", 0)),
                int(arguments.get("bottom", 0)),
            ),
        }
        return self._image_result(payload)

    def _zoom_image(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "ok": True,
            **self.image_session.zoom_image(
                str(arguments.get("image_id", "")),
                float(arguments.get("factor", 0)),
            ),
        }
        return self._image_result(payload)

    def _resize_image(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "ok": True,
            **self.image_session.resize_image(
                str(arguments.get("image_id", "")),
                int(arguments.get("target_width", 0)),
                int(arguments.get("target_height", 0)),
            ),
        }
        return self._image_result(payload)

    def _image_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "output": json.dumps(payload, ensure_ascii=False),
            "error_type": None,
            "generated_image_path": payload["path"],
            "generated_image_id": payload["image_id"],
            "tool_payload": payload,
        }

    @staticmethod
    def _error_result(error_type: str, message: str) -> Dict[str, Any]:
        payload = {"ok": False, "error_type": error_type, "message": message}
        return {
            "output": json.dumps(payload, ensure_ascii=False),
            "error_type": error_type,
            "tool_payload": payload,
        }
