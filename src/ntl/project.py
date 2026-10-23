"""Read, validate and export checked notebook snapshots; no cloud mutations."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil


def repository_root(start=None):
    candidates=[Path(start).resolve()] if start else [Path.cwd().resolve(),Path(__file__).resolve().parent]
    for candidate in candidates:
        for directory in [candidate,*candidate.parents]:
            if (directory/'configs/notebooks.json').is_file() and (directory/'pyproject.toml').is_file():
                return directory
    raise FileNotFoundError('Run from the repository checkout or pass --project PATH')


def profiles(root):
    return json.loads((root/'configs/notebooks.json').read_text(encoding='utf-8'))['profiles']


def project_path(root, relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Profile path leaves the repository')
    return path


def _check_embedded(code, label, depth=0):
    """Compile literal Python modules embedded in the standalone snapshots."""
    if depth>6:raise ValueError('Embedded module nesting exceeds the validation limit')
    compile(code,label,'exec')
    checked=0
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node,ast.Assign):continue
        try:value=ast.literal_eval(node.value)
        except (ValueError,TypeError,SyntaxError,MemoryError,RecursionError):continue
        if isinstance(value,dict):
            for name,text in value.items():
                if isinstance(name,str) and name.endswith('.py') and isinstance(text,str):
                    checked+=1+_check_embedded(text,label+':'+name,depth+1)
        elif isinstance(value,str) and any(isinstance(t,ast.Name) and t.id=='_GAUSSIAN_SOURCE' for t in node.targets):
            checked+=1+_check_embedded(value,label+':gaussian',depth+1)
    return checked


def validate_profile(root, key):
    import nbformat
    profile=profiles(root)[key]
    notebook_path=project_path(root,profile['notebook'])
    metadata=json.loads(project_path(root,profile['metadata']).read_text(encoding='utf-8'))
    digest=hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    if digest!=profile['sha256']:raise ValueError(f'{key}: snapshot checksum differs; record intentional edits in configs/notebooks.json')
    notebook=nbformat.read(notebook_path,as_version=4);nbformat.validate(notebook)
    cells=0;embedded=0
    for index,cell in enumerate(notebook.cells):
        if cell.cell_type!='code':continue
        if cell.outputs or cell.execution_count is not None:raise ValueError(f'{key}: clear notebook outputs before tracking')
        embedded+=_check_embedded(cell.source,f'{key}:cell-{index}')
        cells+=1
    if metadata['code_file']!=notebook_path.name:raise ValueError(f'{key}: code_file and notebook filename differ')
    if not metadata['is_private']:raise ValueError(f'{key}: export template must be private')
    if metadata['session_timeout_seconds']!=32400:raise ValueError(f'{key}: expected a nine-hour session')
    for name in ['dataset_sources','kernel_sources','model_sources','competition_sources']:
        if any(not isinstance(v,str) or not v for v in metadata[name]):raise ValueError(f'{key}: invalid input reference')
    if key=='prepare_llama':
        if metadata['enable_gpu'] or not metadata['enable_internet']:raise ValueError('Llama preparation expects CPU and Internet ON')
    elif metadata['enable_internet'] or not metadata['enable_gpu'] or metadata['machine_shape']!='NvidiaTeslaT4':
        raise ValueError(f'{key}: expected T4 x2 with Internet OFF')
    return {'profile':key,'code_cells':cells,'embedded_modules_compiled':embedded,'sha256':digest}


def scan_source_tree(root):
    """Report filenames and reasons only; never echo possible secret values."""
    excluded={'.git','.venv','venv','__pycache__','.pytest_cache','.ruff_cache','.mypy_cache',
              'artifacts','data','models','outputs','logs','build','dist'}
    text_suffixes={'.py','.json','.toml','.md','.ipynb','.yml','.yaml','.example'}
    patterns={
        'Hugging Face token':re.compile(r'\bhf_[A-Za-z0-9]{25,}\b'),
        'GitHub token':re.compile(r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}\b'),
        'signed cloud URL':re.compile(r'(?i)(?:x-goog-signature|x-amz-signature)\s*='),
        'literal Kaggle credential':re.compile(r'''(?i)(?:KAGGLE_KEY|["']key["'])\s*[:=]\s*["'][a-z0-9_-]{25,}["']'''),
        'machine-specific user path':re.compile(r'(?i)[A-Z]:\\+Users\\+[^\\\s"\']+'),
    }
    findings=[];files=0;total=0
    for path in sorted(root.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(root)
        if set(relative.parts)&excluded or any(p.endswith('.egg-info') for p in relative.parts):continue
        size=path.stat().st_size;files+=1;total+=size
        if size>5_000_000:findings.append({'path':relative.as_posix(),'reason':'file exceeds 5 MB'})
        if path.suffix.lower() in {'.safetensors','.pt','.pth','.bin','.ckpt','.npz','.npy','.parquet','.pkl','.pickle','.joblib'}:
            findings.append({'path':relative.as_posix(),'reason':'model/data/cache artifact outside ignored storage'})
        if path.name.lower().startswith('kaggle') and path.suffix=='.json':
            findings.append({'path':relative.as_posix(),'reason':'Kaggle credential-like filename'})
        if path.suffix not in text_suffixes and path.name not in {'.gitignore','.gitattributes'}:continue
        value=path.read_text(encoding='utf-8')
        for reason,pattern in patterns.items():
            if pattern.search(value):findings.append({'path':relative.as_posix(),'reason':reason})
    return {'files':files,'bytes':total,'findings':findings}


def validate_project(root):
    checked=[validate_profile(root,key) for key in profiles(root)]
    scan=scan_source_tree(root)
    if scan['findings']:raise ValueError(json.dumps({'source_scan_findings':scan['findings']},indent=2))
    return {'passed':True,'profiles':checked,'source_files':scan['files'],'source_bytes':scan['bytes'],
            'network_accessed':False,'GPU_inference_executed':False}


def export_profile(root, key, owner, output=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',owner):raise ValueError('Invalid Kaggle username')
    validate_profile(root,key)
    profile=profiles(root)[key]
    notebook=project_path(root,profile['notebook'])
    metadata=json.loads(project_path(root,profile['metadata']).read_text(encoding='utf-8'))
    metadata['id']=owner+'/'+metadata['id'].split('/',1)[1]
    destination=Path(output).resolve() if output else root/'artifacts/kaggle'/key
    if destination.resolve()==root.resolve():raise ValueError('Choose an export directory below artifacts or outside the repository root')
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(notebook,destination/notebook.name)
    (destination/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return {'profile':key,'directory':str(destination),'kernel':metadata['id'],
            'competition_submission':profile['competition_submission'],'cloud_action_performed':False}
