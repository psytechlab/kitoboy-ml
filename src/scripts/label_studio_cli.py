"""
The scipt allow to communicate with Label Studio instance.
It can deploy the project given the labeling interface and data to be annotated 
as well as download annotated data either in csv or json
"""
import argparse
import json
import io
import pandas as pd
import requests

import yaml

import os
import sys

sys.path.append(os.getcwd())

from src.utils.utils import save_to_json, load_json

settings = {}


def get_name2id(args: argparse.Namespace, token: str) -> dict[str, str]:
    """Get the dict of projects and its corresponding id from Ladel Studio.

    Args:
        args: ArgParse object.
        token: Access token for Label Studio.
    Returns:
        Dictionary with project names and its id.
    """
    r = requests.get(f'{settings["host"]}:{args.port}/api/projects/', headers={'Authorization': f'Token {token}'})
    project_name2id = {x['title']: x['id'] for x in r.json()['results']}
    return project_name2id


def get_annotation_from_ls(args: argparse.Namespace, token: str) -> None:
    """Download the annotation from Ladel Studio project.

    Args:
        args: ArgParse object.
        token: Access token for Label Studio.
    Returns:
        None
    """
    project_name2id = get_name2id(args, token)
    project_id = project_name2id[args.project_name]
    r = requests.get(f'{settings["host"]}:{args.port}/api/projects/{project_id}/export?exportType=JSON',
                     headers={'Authorization': f'Token {token}'}, )
    data = r.json()
    if args.store_source_json:
        save_to_json(data, args.output_file)
        return
    annotated_data = []
    for d in data:
        ls_id = d['id']
        data_id = d['data'].get('data_id')
        if data_id is None:
            data_id = d['data'].get('id', -1)
        text = d['data']['text']
        for annot in d['annotations']:
            complited_by = annot['completed_by']
            was_canceled = annot['was_cancelled']
            comment = ''
            checkboxes = ''
            for res in annot['result']:
                if res['type'] == 'choices':
                    checkboxes = ';'.join(res['value']['choices'])
                elif res['type'] == 'textarea':
                    comment = ' '.join(res['value']['text'])
        annotated_data.append((ls_id, data_id, text, complited_by, args.port, was_canceled, checkboxes, comment))
    annotated_data = pd.DataFrame(annotated_data, columns=[
        'ls_id', 'data_id', 'text', 'complited_by', 'port', 'was_canceled', 'annotation', 'comment'
    ])
    if args.with_comments_only:
        annotated_data = annotated_data[annotated_data.comment != '']
    annotated_data.to_csv(args.output_file)


def deploy_project_in_ls(args: argparse.Namespace, token: str) -> None:
    """Deploy the annotation project in Ladel Studio.

    Args:
        args: ArgParse object.
        token: Access token for Label Studio.
    Returns:
        None
    """
    project_name2id = get_name2id(args, token)
    if args.project_name in project_name2id:
        raise ValueError(f"The project with a name '{args.project_name}' already exists") 
    with open(args.interface_config) as f:
        label_config = f.read()

    if args.task_file.endswith('csv'):
        upload_data = pd.read_csv(args.task_file)
        upload_data = upload_data.rename({"sent":"text"}, axis=1)
    elif args.task_file.endswith('json'):
        upload_dat = load_json(f)
    else:
        raise ValuerError

    s = io.StringIO()

    if isinstance(upload_data, pd.DataFrame):
        upload_data.to_csv(s)
    elif isinstance(upload_data, list):
        json.dump(upload_data, s)
    else:
        raise ValueError

    upload_data = s.getvalue()
    r = requests.post(
        f'{settings["host"]}:{args.port}/api/projects/',
        headers={'Authorization': f'Token {token}'},
        json={'title': args.project_name, 'label_config': label_config}
    )
    created_project_id = r.json()['id']
    r = requests.post(
        f'{settings["host"]}:{args.port}/api/projects/{created_project_id}/import',
        headers={'Authorization': f'Token {token}'},
        files={args.task_file.split('/')[-1]: upload_data}
    )


def main(args: argparse.Namespace) -> None:
    """Main function.

    Args:
        args: ArgParse object.
    Returns:
        None
    """
    with open(args.settings) as f:
        global settings
        settings = yaml.safe_load(f)
    machines = settings['machines']
    if args.port not in machines:
        raise KeyError('Unknown machine. Check the port number')
    token = machines[args.port]
    if args.command == 'get_annotation':
        get_annotation_from_ls(args, token)
    elif args.command == 'deploy_project':
        deploy_project_in_ls(args, token)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--settings',
        type=str,
        default='reference/settings.yaml',
        help='The path to yaml file with settings'
    )
    parser.add_argument(
        '--port',
        '-p',
        type=int,
        required=True,
        help='The Label studio instance port'
    )
    parser.add_argument(
        '--project_name',
        type=str,
        required=True,
        help='The project name to work with'
    )
    subparsers = parser.add_subparsers(
        dest='command',
        help='Define whether to download or deploy the annotation'
    )

    get_annotation_parser = subparsers.add_parser(
        'get_annotation',
        help='Download the annotation from Label Studio'
    )
    get_annotation_parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='The file name where to store the downloaded annotation'
    )
    get_annotation_parser.add_argument(
        '--with_comments_only',
        action='store_true',
        help='Save the annotation with comments only'
    )
    get_annotation_parser.add_argument(
        '--store_source_json',
        action='store_true',
        help='Store annotation in Label studio json format'
    )

    deploy_iteration_parser = subparsers.add_parser(
        'deploy_project',
        help='Deploy the annotation project'
    )
    deploy_iteration_parser.add_argument(
        '--task_file',
        type=str,
        help='The file with annotation either in csv or json'
    )
    deploy_iteration_parser.add_argument(
        '--interface_config',
        type=str, help='Path to the config with label interface'
    )

    args = parser.parse_args()
    main(args)
