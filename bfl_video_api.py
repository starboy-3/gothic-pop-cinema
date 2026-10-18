import json
from pathlib import Path
import shutil
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen


API = 'https://api.bfl.ai/v1'


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def api_request(url, key, payload=None):
    address = urlsplit(url)
    host = address.hostname or ''
    if (address.scheme != 'https' or address.username or address.password
            or not (host == 'api.bfl.ai' or host.endswith('.bfl.ai'))):
        raise ValueError('Untrusted API host')
    request = Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'x-key': key, 'Content-Type': 'application/json'},
    )
    try:
        with build_opener(NoRedirect()).open(request, timeout=120) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors='replace').replace(key, '[REDACTED]')
        raise RuntimeError(f'BFL HTTP {error.code}: {detail[:1200]}') from None


def download(url, destination):
    if urlsplit(url).scheme != 'https':
        raise ValueError('Expected HTTPS artifact URL')
    destination = Path(destination)
    temporary = destination.with_suffix('.part')
    with urlopen(url, timeout=180) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size == 0:
        raise RuntimeError('Empty artifact')
    temporary.replace(destination)
