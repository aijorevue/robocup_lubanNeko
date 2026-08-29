"""Hash-checked atomic replacement of a staged Python file on the current host."""
import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def deploy(target, staged, expected, candidate, commit=False):
    target = Path(target).resolve(strict=True)
    staged = Path(staged).resolve(strict=True)
    if target == staged or target.parent != staged.parent:
        raise ValueError('Staged file must be distinct and in the canonical target directory')
    if not target.is_file() or not staged.is_file():
        raise ValueError('Target and staged candidate must be regular files')
    if target.stat().st_dev != staged.stat().st_dev:
        raise ValueError('Target and candidate must use the same filesystem')
    if digest(target) != expected or digest(staged) != candidate:
        raise ValueError('Hash mismatch; preserve concurrent edits and recheck the source')
    compile(staged.read_bytes(), str(target), 'exec')
    receipt = {'target': str(target), 'staged': str(staged),
               'before_sha256': expected, 'after_sha256': candidate, 'committed': False}
    if not commit:
        return receipt

    lock_path = target.with_name('.' + target.name + '.rk-deploy.lock')
    lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(lock_fd, json.dumps({'pid': os.getpid(), 'target': str(target)}).encode())
        os.fsync(lock_fd)
        if digest(target) != expected or digest(staged) != candidate:
            raise ValueError('Source or stage changed before commit')
        stat = target.stat()
        backup = target.with_name('.' + target.name + '.rk-backup-' + uuid.uuid4().hex + '.bak')
        with target.open('rb') as source, backup.open('xb') as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        shutil.copystat(target, backup)
        if digest(backup) != expected:
            raise ValueError('Backup verification failed')
        receipt['backup'] = str(backup)
        os.chmod(staged, stat.st_mode & 0o777)
        if hasattr(os, 'chown'):
            os.chown(staged, stat.st_uid, stat.st_gid)
        # Windows _commit requires a writable descriptor; opening r+b also
        # works on POSIX and does not alter the candidate bytes.
        with staged.open('r+b') as source:
            os.fsync(source.fileno())
        if digest(target) != expected or digest(staged) != candidate:
            raise ValueError('Concurrent source or stage change detected; replacement cancelled')
        os.replace(staged, target)
        receipt['committed'] = True
        if hasattr(os, 'O_DIRECTORY'):
            directory_fd = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        if digest(target) != candidate:
            raise RuntimeError('Post-replacement hash mismatch; inspect and restore the backup')
        return receipt
    finally:
        os.close(lock_fd)
        lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--staged', required=True, type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--commit', action='store_true')
    args = parser.parse_args()
    print(json.dumps(deploy(args.target, args.staged, args.expected_sha256,
                            args.candidate_sha256, args.commit), indent=2))


if __name__ == '__main__':
    main()
