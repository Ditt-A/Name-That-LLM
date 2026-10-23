"""Offline project commands. Exporting never runs or submits a Kaggle notebook."""
import argparse
import json

from .project import export_profile,profiles,repository_root,validate_project


def main(argv=None):
    parser=argparse.ArgumentParser(prog='ntl')
    parser.add_argument('--project',help='Path to the repository checkout')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='List notebook profiles')
    sub.add_parser('validate',help='Validate snapshots, metadata, embedded code and source hygiene offline')
    export=sub.add_parser('export',help='Create a local Kaggle upload directory')
    export.add_argument('profile');export.add_argument('--owner',required=True)
    export.add_argument('--output',help='Optional destination; defaults to artifacts/kaggle/PROFILE')
    args=parser.parse_args(argv)
    try:
        root=repository_root(args.project)
        if args.command=='list':
            result={key:{'description':p['description'],'submission':p['competition_submission']}
                    for key,p in profiles(root).items()}
        elif args.command=='validate':result=validate_project(root)
        else:result=export_profile(root,args.profile,args.owner,args.output)
        print(json.dumps(result,indent=2));return 0
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f'{error}\n')
