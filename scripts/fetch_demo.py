"""Fetch one JAAD clip using ZIP byte ranges; do not download the 3 GB archive."""
import io
import json
import hashlib
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip'
ROOT = Path(__file__).resolve().parents[1]


class RemoteArchive(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r:
            self.size = int(r.headers['Content-Length'])

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        return self.position

    def tell(self):
        return self.position

    def read(self, n=-1):
        n = min(n if n >= 0 else self.size-self.position, self.size-self.position)
        if n <= 0:
            return b''
        if n > 100_000_000:
            raise ValueError('Refusing unexpectedly large clip download')
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.position}-{self.position+n-1}'})
        with urllib.request.urlopen(request, timeout=120) as r:
            if r.status != 206:
                raise RuntimeError('Server did not honor partial download')
            data = r.read(n)
        if len(data) != n:
            raise RuntimeError('Incomplete range response')
        self.position += len(data)
        return data


if __name__ == '__main__':
    output = ROOT/'data/raw/demo'
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RemoteArchive(URL)) as archive:
        member = next(i for i in archive.infolist() if i.filename.endswith('/video_0005.mp4') or i.filename == 'video_0005.mp4')
        data = archive.read(member)
    (output/'video_0005.mp4').write_bytes(data)
    (output/'source.json').write_text(json.dumps({'source': URL, 'member': member.filename,
        'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
        'attribution': 'Rasouli, Kotseruba and Tsotsos, JAAD, ICCVW 2017',
        'video_license': 'CC BY 4.0 per JAAD 2.0 README; original webpage also refers to MIT'}, indent=2))
    print('Saved permitted JAAD clip:', len(data), 'bytes')
