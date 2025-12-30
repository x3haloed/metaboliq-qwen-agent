import csv
import os
from typing import Any, Dict, List

from qwen_agent.tools.shape_handlers.base import ShapeHandler
from qwen_agent.tools.shape_handlers.common import paginate_list

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None


class TableHandler(ShapeHandler):
    kind = 'table'
    extensions = {'.csv', '.tsv'}

    def outline(self, path: str, page: int, page_size: int) -> Dict[str, Any]:
        if pd is not None:
            df = pd.read_csv(path)
            page_info = paginate_list(list(range(len(df))), page, page_size)
            start = (page_info['page'] - 1) * page_info['page_size']
            end = start + page_info['page_size']
            outline = {
                'summary': 'table',
                'row_count': len(df),
                'columns': list(df.columns),
                'head': df.iloc[start:end].to_dict(orient='records'),
            }
            outline.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
            if page_info['truncated']:
                outline['note'] = (
                    f'Rows truncated. Call describe_file with page={page_info["next_page"]} '
                    f'to continue.'
                )
            return outline
        header, rows = self._read_table(path)
        page_info = paginate_list(rows, page, page_size)
        outline = {
            'summary': 'table',
            'row_count': len(rows),
            'columns': header,
            'head': page_info['items'],
        }
        outline.update({k: page_info[k] for k in ('page', 'page_size', 'total', 'truncated', 'next_page')})
        if page_info['truncated']:
            outline['note'] = (
                f'Rows truncated. Call describe_file with page={page_info["next_page"]} '
                f'to continue.'
            )
        return outline

    def select(self, path: str, selector: List[Any], page: int, page_size: int) -> Dict[str, Any]:
        if not isinstance(selector, list) or len(selector) != 2:
            raise ValueError('Table selector must be [row_index, column]')
        row, col = selector
        if pd is not None:
            df = pd.read_csv(path)
            if isinstance(col, int):
                col = df.columns[col]
            try:
                return {'value': df.at[row, col]}
            except Exception as exc:
                raise KeyError(f'No section found at {selector!r}: {exc}') from exc
        header, rows = self._read_table(path)
        if isinstance(col, int):
            col_idx = col
        else:
            col_idx = header.index(col)
        try:
            return {'value': rows[row][col_idx]}
        except Exception as exc:
            raise KeyError(f'No section found at {selector!r}: {exc}') from exc

    def replace(self, path: str, selector: List[Any], value: Any) -> Dict[str, Any]:
        if not isinstance(selector, list) or len(selector) != 2:
            raise ValueError('Table selector must be [row_index, column]')
        row, col = selector
        if pd is not None:
            df = pd.read_csv(path)
            if isinstance(col, int):
                col = df.columns[col]
            try:
                df.at[row, col] = value
            except Exception as exc:
                raise KeyError(f'No section found at {selector!r}: {exc}') from exc
            df.to_csv(path, index=False)
            return {'changed': True, 'kind': self.kind}
        header, rows = self._read_table(path)
        if isinstance(col, int):
            col_idx = col
        else:
            col_idx = header.index(col)
        try:
            rows[row][col_idx] = value
        except Exception as exc:
            raise KeyError(f'No section found at {selector!r}: {exc}') from exc
        self._write_table(path, header, rows)
        return {'changed': True, 'kind': self.kind}

    @staticmethod
    def _read_table(path: str):
        ext = os.path.splitext(path)[1].lower()
        delimiter = '\t' if ext == '.tsv' else ','
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
        if not rows:
            return [], []
        return rows[0], rows[1:]

    @staticmethod
    def _write_table(path: str, header: List[str], rows: List[List[str]]) -> None:
        ext = os.path.splitext(path)[1].lower()
        delimiter = '\t' if ext == '.tsv' else ','
        with open(path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(header)
            writer.writerows(rows)
