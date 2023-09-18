#! /usr/bin/env python

import os
import sys
import copy
import math
import tqdm
import onnx
import onnxruntime as ort
import onnx_graphsurgeon as gs
import numpy as np
import subprocess
import pkg_resources
from rich import print as rich_print
from rich.table import Table
from rich.text import Text
from collections import defaultdict
from argparse import ArgumentParser
from typing import Optional, Dict, List, Any, Callable
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

DEFAULT_ONNXSIM_VER = '0.4.33'

OPERATIONS_TO_BE_OPTIMIZED = [
    'Einsum',
    'OneHot',
]

ONNX_DTYPES_TO_NUMPY_DTYPES: dict = {
    f'{onnx.TensorProto.FLOAT16}': np.float16,
    f'{onnx.TensorProto.FLOAT}': np.float32,
    f'{onnx.TensorProto.DOUBLE}': np.float64,
    f'{onnx.TensorProto.INT8}': np.int8,
    f'{onnx.TensorProto.INT16}': np.int16,
    f'{onnx.TensorProto.INT32}': np.int32,
    f'{onnx.TensorProto.INT64}': np.int64,
    f'{onnx.TensorProto.UINT8}': np.uint8,
    f'{onnx.TensorProto.UINT16}': np.uint16,
    f'{onnx.TensorProto.UINT32}': np.uint32,
    f'{onnx.TensorProto.UINT64}': np.uint64,
}


def get_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return ''

def install_package(package_name):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def human_readable_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:,.1f}Yi{suffix}"

def parse_shapes(shapes_arg) -> Dict:
    shapes: Dict = {}
    if shapes_arg is not None:
        for x in shapes_arg:
            if ':' not in x:
                shapes[None] = list(map(int, x.split(',')))
            else:
                pieces = x.split(':')
                # for the input name like input:0
                name, shape = ':'.join(
                    pieces[:-1]), list(map(int, pieces[-1].split(',')))
                shapes.update({name: shape})
    return shapes

def dummy_onnx_inference(
    *,
    onnx_graph: onnx.ModelProto,
    output_names: List[str],
) -> List[np.ndarray]:
    """Perform inference on ONNX subgraphs with an all-1 dummy tensor.

    Parameters
    ----------
    onnx_graph: onnx.ModelProto
        ONNX subgraphs

    output_names: List[str]
        List of output names to be checked for output values

    Returns
    ----------
    outputs: List[np.ndarray]
        Results of inference using dummy tensor
    """
    # Separate onnx at specified output_names position
    gs_graph = gs.import_onnx(onnx_graph)

    # reduce all axes except batch axis
    for i, node in enumerate(gs_graph.nodes):
        if gs_graph.opset <= 17 \
            and gs_graph.nodes[i].op in ['ReduceMax', 'ReduceMean', 'ReduceMin', 'ReduceProd'] \
            and 'axes' not in node.attrs:
            gs_graph.nodes[i].attrs['axes'] = [
                i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))
            ] if len(gs_graph.nodes[i].inputs[0].shape) > 1 else [0]

        elif gs_graph.opset > 17 \
            and gs_graph.nodes[i].op in ['ReduceMax', 'ReduceMean', 'ReduceMin', 'ReduceProd'] \
            and len(gs_graph.nodes[i].inputs) == 1:
            const_axes = [
                i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))
            ] if len(gs_graph.nodes[i].inputs[0].shape) > 1 else [0]
            gs_graph.nodes[i].inputs.append(
                gs.Constant(
                    f'{gs_graph.nodes[i].name}_axes',
                    values=np.asarray(const_axes, dtype=np.int64)
                )
            )

        elif gs_graph.opset <= 12 \
            and gs_graph.nodes[i].op in ['ReduceSum'] \
            and 'axes' not in node.attrs:
            gs_graph.nodes[i].attrs['axes'] = [
                i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))
            ] if len(gs_graph.nodes[i].inputs[0].shape) > 1 else [0]

        elif gs_graph.opset > 12 \
            and gs_graph.nodes[i].op in ['ReduceSum'] \
            and len(gs_graph.nodes[i].inputs) == 1:
            const_axes = [
                i for i in range(1, len(gs_graph.nodes[i].inputs[0].shape))
            ] if len(gs_graph.nodes[i].inputs[0].shape) > 1 else [0]
            gs_graph.nodes[i].inputs.append(
                gs.Constant(
                    f'{gs_graph.nodes[i].name}_axes',
                    values=np.asarray(const_axes, dtype=np.int64)
                )
            )

    # instead, modify onnx graph manually
    gs_graph.outputs = []
    for graph_node in gs_graph.nodes:
        for node_output in graph_node.outputs:
            if node_output.name in output_names:
                if node_output.dtype is not None:
                    gs_graph.outputs.append(node_output)

    new_onnx_graph = gs.export_onnx(graph=gs_graph, do_type_check=False)
    tmp_onnx_path = ''
    tmp_onnx_external_weights_path =''
    try:
        serialized_graph = onnx._serialize(new_onnx_graph)
    except ValueError as ve:
        tmp_onnx_path = 'tmp.onnx'
        tmp_onnx_external_weights_path ='tmp_external.weights'
        onnx.save(
            proto=new_onnx_graph,
            f=tmp_onnx_path,
            save_as_external_data=True,
            location=tmp_onnx_external_weights_path
        )
        serialized_graph = tmp_onnx_path
    sess_options = ort.SessionOptions()
    onnx_session = ort.InferenceSession(
        path_or_bytes=serialized_graph,
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    onnx_inputs = gs_graph.inputs
    input_names: List[str] = [inp.name for inp in onnx_inputs]
    input_sizes: List[int] = [inp.shape for inp in onnx_inputs]
    new_input_sizes = []
    for input_size in input_sizes:
        new_input_size = []
        for idx, dim in enumerate(input_size):
            if idx == 0 and input_sizes[0][0] is not None \
                and not isinstance(input_sizes[0][0], str) \
                and len(input_sizes[0]) == len(input_size) \
                and (dim is None or isinstance(dim, str)):
                # Batch size assignment for input OPs
                new_input_size.append(input_sizes[0][0])
            elif dim is None or isinstance(dim, str):
                # Fixed and assigned 1
                new_input_size.append(1)
            else:
                # Assign input shape as is
                new_input_size.append(dim)
        new_input_sizes.append(new_input_size)
    input_sizes = new_input_sizes
    input_dtypes: List[Any] = [inp.dtype for inp in onnx_inputs]
    input_datas = {}

    for input_name, input_size, input_dtype in zip(input_names, input_sizes, input_dtypes):
        input_datas[input_name] = np.ones(
            input_size,
            dtype=input_dtype,
        )

    outputs = onnx_session.run(None, input_datas)
    if tmp_onnx_path:
        os.remove(tmp_onnx_path)
        os.remove(tmp_onnx_external_weights_path)
    return outputs

def print_simplifying_info(model_ori: onnx.ModelProto, model_opt: onnx.ModelProto) -> None:
    """
    Cited: https://github.com/daquexian/onnx-simplifier/blob/ae8b749eb19ea2e3a85e364a91e299bacb690b0e/onnxsim/model_info.py#L57-L88
    --------------------------------------------------------
    |             | original model | simplified model |
    --------------------------------------------------------
    | ****        | ****           | ****             |
    --------------------------------------------------------
    | Model Size  | ****           | ****             |
    --------------------------------------------------------
    """
    ori_info = ModelInfo(model_ori)
    opt_info = ModelInfo(model_opt)
    table = Table()
    table.add_column('')
    table.add_column('Original Model')
    table.add_column('Simplified Model')

    def add_row(table: Table, key, ori_data, opt_data, is_better: Callable[[Any, Any], Any], postprocess: Optional[Callable[[Any], Any]] = None) -> None:
        if postprocess is None:
            postprocess = str
        if is_better(opt_data, ori_data):
            table.add_row(key, postprocess(ori_data), Text(postprocess(opt_data), style='bold green1'))
        else:
            table.add_row(key, postprocess(ori_data), postprocess(opt_data))

    for key in sorted(list(set(ori_info.op_nums.keys()) | set(opt_info.op_nums.keys()))):
        add_row(table, key, f"{ori_info.op_nums[key]:,}", f"{opt_info.op_nums[key]:,}", lambda opt, ori: opt < ori)

    table.add_row('----------------------', '----------------', '----------------')
    ori_ops_count = sum([ori_info.op_nums[key] for key in ori_info.op_nums.keys()])
    opt_ops_count = sum([opt_info.op_nums[key] for key in opt_info.op_nums.keys()])
    table.add_row('Total number of OPs', f"{ori_ops_count:,}", f"{opt_ops_count:,}")
    table.add_row('======================', '================', '================')
    add_row(table, 'Model Size', ori_info.model_size, opt_info.model_size, lambda opt, ori: opt < ori, postprocess=human_readable_size)
    rich_print(table)

class ModelInfo:
    """
    Model info contains:
    1. Num of every op
    2. Model size
    """
    def __init__(self, model: onnx.ModelProto):
        self.op_nums = defaultdict(int)
        for node in model.graph.node:
            self.op_nums[node.op_type] += 1
        self.model_size = model.ByteSize()


def partial_optimization(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_onnx_file_path: Optional[str] = '',
    overwrite_input_shape: Optional[Dict] = None,
    optimization_times: Optional[int] = 0,
    target_onnxsim_version: Optional[str] = '0.4.30',
    non_verbose: Optional[bool] = False,
) -> onnx.ModelProto:
    """

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_onnx_file_path: Optional[str]
        Output onnx file path.\n\
        If not specified, it will overwrite the onnx specified in --input_onnx_file_path.

    overwrite_input_shape: Optional[Dict]
        Overwrite the input shape.\n\
        The format is\n\
        {'data1': [1, 3, 224, 224], 'data2': [1, 224], 'data3': [1]}

    optimization_times: Optional[int]
        Number of times the optimization process is performed.\n\
        If zero is specified, the tool automatically calculates the number of optimization times.\n\
        Default: 0

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.

    Returns
    -------
    onnx_graph: onnx.ModelProto
        Optimized onnx ModelProto
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # overwrite_input_shape
    if overwrite_input_shape is not None \
        and not isinstance(overwrite_input_shape, Dict):
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'overwrite_input_shape must be specified by list.'
        )
        sys.exit(1)

    # optimization_times
    if optimization_times is not None \
        and optimization_times < 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'optimization_times must be 1 or more.'
        )
        sys.exit(1)

    # Get the version number of the onnxsim
    prev_onnxsim_ver: str = get_version('onnxsim')
    if not prev_onnxsim_ver:
        prev_onnxsim_ver = DEFAULT_ONNXSIM_VER

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)

    # import
    gs_graph = gs.import_onnx(onnx_graph)

    # optimization_times
    if optimization_times == 0:
        optimization_times = math.ceil(len(gs_graph.nodes) / 1000)

    # Check the shape of the input tensor
    # If undefined dimension exists and overwrite_input_shape is not specified,
    # an error is generated.
    for gs_graph_input in gs_graph.inputs:
        number_of_undefined_dimensions = \
            sum([1 if isinstance(s, str) else 0 for s in gs_graph_input.shape])
        if overwrite_input_shape is None and number_of_undefined_dimensions > 0:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'To optimize onnx with undefined dimensions, '+
                f'specify --overwrite_input_shape to fix the shape of the input tensor.'
            )
            sys.exit(1)

    # Initialization of useless output shapes
    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Optimization of graph output shape in progress.')
    try:
        output_clear = False
        for graph_output in gs_graph.outputs:
            if graph_output.shape is not None \
                and sum([1 if isinstance(s, int) and s < 1 else 0 for s in graph_output.shape]) > 0:
                graph_output.shape = None
                output_clear = True
        if output_clear:
            onnx_graph_opt = onnx.shape_inference.infer_shapes(gs.export_onnx(gs_graph, do_type_check=False))
            if input_onnx_file_path:
                onnx.save(onnx_graph_opt, f=input_onnx_file_path)
        else:
            onnx_graph_opt = onnx_graph
    except:
        onnx_graph_opt = onnx_graph

    # Force installation of onnxsim==0.4.30
    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} onnxsim=={target_onnxsim_version} being installed.')
    install_package(f'onnxsim=={target_onnxsim_version}')
    from onnxsim import simplify

    # Extract OPs to be optimized
    target_ops: List[gs.Node] = [
        gs_graph_node \
            for gs_graph_node in gs_graph.nodes \
                if gs_graph_node.op in OPERATIONS_TO_BE_OPTIMIZED
    ]
    target_ops_names: List[str] = [
        target_op.name for target_op in target_ops
    ]

    # If target_ops is zero case, just run onnxsim and exit.
    # If target_ops is one or more, overwrite the output shape of the target OP
    # and execute onnxsim to terminate the process.
    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Model optimization in progress. If there are thousands of OPs, it will take several minutes.')

    if len(target_ops) == 0:
        # Simple onnxsim
        model_simp = copy.deepcopy(onnx_graph_opt)
        try:
            for _ in tqdm.tqdm(range(optimization_times), dynamic_ncols=True, disable=non_verbose):
                model_simp, check = simplify(
                    model=model_simp,
                    overwrite_input_shapes=overwrite_input_shape,
                )
            onnx_graph_opt = model_simp
        except:
            pass
        finally:
            del model_simp

    else:
        # Custom onnxsim

        # ONNX dummy inference
        # Generate output for all OPs. Used to verify the output error of each OP in the TensorFlow model.
        full_ops_output_names = []
        onnx_tensor_infos_for_validation = None
        for graph_node in gs_graph.nodes:
            if graph_node.name in target_ops_names:
                full_ops_output_names_sub = []
                for graph_node_output in graph_node.outputs:
                    full_ops_output_names_sub.append(graph_node_output.name)
                full_ops_output_names.extend(full_ops_output_names_sub)
        # Models with errors during inference in onnxruntime skip dummy inference.
        try:
            onnx_outputs_for_validation: List[np.ndarray] = \
                dummy_onnx_inference(
                    onnx_graph=onnx_graph_opt,
                    output_names=full_ops_output_names,
                )
            """
            onnx_tensor_infos_for_validation:
                {
                    onnx_output_name: np.ndarray,
                    onnx_output_name: np.ndarray,
                    onnx_output_name: np.ndarray,
                                :
                }
            """
            onnx_tensor_infos_for_validation = {
                ops_output_name: onnx_output_for_validation \
                    for ops_output_name, onnx_output_for_validation \
                        in zip(full_ops_output_names, onnx_outputs_for_validation)
            }
            del onnx_outputs_for_validation
        except Exception as ex:
            print(f'{ex}')

        # Addressing Einsum and OneHot shape_inference failure for onnx.
        if onnx_tensor_infos_for_validation is not None:
            model_simp = copy.deepcopy(onnx_graph_opt)
            try:
                model_simp, check = simplify(
                    model=model_simp,
                    overwrite_input_shapes=overwrite_input_shape,
                )
            except:
                pass
            for _ in tqdm.tqdm(range(optimization_times), dynamic_ncols=True, disable=non_verbose):
                gs_graph = gs.import_onnx(model_simp)
                target_ops: List[gs.Node] = [
                    gs_graph_node \
                        for gs_graph_node in gs_graph.nodes \
                            if gs_graph_node.op in OPERATIONS_TO_BE_OPTIMIZED
                ]
                target_ops_names: List[str] = [
                    target_op.name for target_op in target_ops
                ]
                for target_op in target_ops:
                    correction_op_output: gs.Variable = target_op.outputs[0]
                    if correction_op_output.name in onnx_tensor_infos_for_validation:
                        onnx_output_shape = list(onnx_tensor_infos_for_validation[correction_op_output.name].shape)
                        correction_op_output.shape = onnx_output_shape
                try:
                    model_simp = gs.export_onnx(gs_graph)
                    model_simp, check = simplify(
                        model=model_simp,
                        overwrite_input_shapes=overwrite_input_shape,
                    )
                except:
                    pass
                onnx_graph_opt = model_simp
            del model_simp

    # Reinstall the original onnxsim version
    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} The version of onnxsim is being restored.')
    install_package(f'onnxsim=={prev_onnxsim_ver}')

    # Display of optimization results
    if not non_verbose:
        print_simplifying_info(onnx_graph, onnx_graph_opt)

    if output_onnx_file_path:
        onnx.save(proto=onnx_graph, f=output_onnx_file_path)
        if not non_verbose:
            print(f'{Color.GREEN}INFO:{Color.RESET} Save onnx to: {output_onnx_file_path}')
    elif input_onnx_file_path:
        onnx.save(proto=onnx_graph, f=input_onnx_file_path)
        if not non_verbose:
            print(f'{Color.GREEN}INFO:{Color.RESET} Save onnx to: {input_onnx_file_path}')

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

    return onnx_graph


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-if',
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '-of',
        '--output_onnx_file_path',
        type=str,
        help=\
            'Output onnx file path. \n'+
            'If not specified, it will overwrite the onnx specified in --input_onnx_file_path.'
    )
    parser.add_argument(
        '-ois',
        '--overwrite_input_shape',
        type=str,
        nargs='+',
        help=\
            'Overwrite the input shape. \n' +
            'The format is\n' +
            '"input_name_1:dim0,...,dimN" "input_name_2:dim0,...,dimN" "input_name_3:dim0,...,dimN". \n' +
            'When there is only one input, for example, \n' +
            '"data:1,3,224,224" \n' +
            'When there are multiple inputs, for example, \n' +
            '"data1:1,3,224,224" "data2:1,3,112,112" "data3:5" \n' +
            'A value of 1 or more must be specified. \n' +
            'Numerical values other than dynamic dimensions are ignored.'
    )
    parser.add_argument(
        '-ot',
        '--optimization_times',
        type=int,
        default=0,
        help=\
            'Number of times the optimization process is performed. \n' +
            'If zero is specified, the tool automatically calculates the number of optimization times. \n' +
            'Default: 0'
    )
    parser.add_argument(
        '-tov',
        '--target_onnxsim_version',
        type=str,
        default='0.4.30',
        help=\
            'Version number of the onnxsim used for optimization. \n'+
            'Default: 0.4.30'
    )
    parser.add_argument(
        '-n',
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    input_onnx_file_path: str = args.input_onnx_file_path
    output_onnx_file_path: str = args.output_onnx_file_path
    overwrite_input_shape = args.overwrite_input_shape
    if overwrite_input_shape:
        overwrite_input_shape: Dict = parse_shapes(overwrite_input_shape)
    optimization_times: int = args.optimization_times
    target_onnxsim_version: str = args.target_onnxsim_version
    non_verbose: bool = args.non_verbose

    # structure check
    partial_optimization(
        input_onnx_file_path=input_onnx_file_path,
        output_onnx_file_path=output_onnx_file_path,
        overwrite_input_shape=overwrite_input_shape,
        optimization_times=optimization_times,
        target_onnxsim_version= target_onnxsim_version,
        non_verbose=non_verbose,
    )


if __name__ == '__main__':
    main()