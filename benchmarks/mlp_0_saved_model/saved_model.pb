ܸ
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028��
�
*adam/Sequential_mlp_mlp_0weight_0_velocityVarHandleOp*
_output_shapes
: *;

debug_name-+adam/Sequential_mlp_mlp_0weight_0_velocity/*
dtype0*
shape:	�
*;
shared_name,*adam/Sequential_mlp_mlp_0weight_0_velocity
�
>adam/Sequential_mlp_mlp_0weight_0_velocity/Read/ReadVariableOpReadVariableOp*adam/Sequential_mlp_mlp_0weight_0_velocity*
_output_shapes
:	�
*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOp*adam/Sequential_mlp_mlp_0weight_0_velocity*
_class
loc:@Variable*
_output_shapes
:	�
*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:	�
*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
f
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:	�
*
dtype0
�
*adam/Sequential_mlp_mlp_0weight_0_momentumVarHandleOp*
_output_shapes
: *;

debug_name-+adam/Sequential_mlp_mlp_0weight_0_momentum/*
dtype0*
shape:	�
*;
shared_name,*adam/Sequential_mlp_mlp_0weight_0_momentum
�
>adam/Sequential_mlp_mlp_0weight_0_momentum/Read/ReadVariableOpReadVariableOp*adam/Sequential_mlp_mlp_0weight_0_momentum*
_output_shapes
:	�
*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOp*adam/Sequential_mlp_mlp_0weight_0_momentum*
_class
loc:@Variable_1*
_output_shapes
:	�
*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:	�
*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	�
*
dtype0
�
(adam/Sequential_mlp_mlp_0bias_0_velocityVarHandleOp*
_output_shapes
: *9

debug_name+)adam/Sequential_mlp_mlp_0bias_0_velocity/*
dtype0*
shape:
*9
shared_name*(adam/Sequential_mlp_mlp_0bias_0_velocity
�
<adam/Sequential_mlp_mlp_0bias_0_velocity/Read/ReadVariableOpReadVariableOp(adam/Sequential_mlp_mlp_0bias_0_velocity*
_output_shapes
:
*
dtype0
�
%Variable_2/Initializer/ReadVariableOpReadVariableOp(adam/Sequential_mlp_mlp_0bias_0_velocity*
_class
loc:@Variable_2*
_output_shapes
:
*
dtype0
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape:
*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:
*
dtype0
�
(adam/Sequential_mlp_mlp_0bias_0_momentumVarHandleOp*
_output_shapes
: *9

debug_name+)adam/Sequential_mlp_mlp_0bias_0_momentum/*
dtype0*
shape:
*9
shared_name*(adam/Sequential_mlp_mlp_0bias_0_momentum
�
<adam/Sequential_mlp_mlp_0bias_0_momentum/Read/ReadVariableOpReadVariableOp(adam/Sequential_mlp_mlp_0bias_0_momentum*
_output_shapes
:
*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOp(adam/Sequential_mlp_mlp_0bias_0_momentum*
_class
loc:@Variable_3*
_output_shapes
:
*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:
*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
e
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:
*
dtype0
�
Sequential_mlp_mlp_0biasVarHandleOp*
_output_shapes
: *)

debug_nameSequential_mlp_mlp_0bias/*
dtype0*
shape:
*)
shared_nameSequential_mlp_mlp_0bias
�
,Sequential_mlp_mlp_0bias/Read/ReadVariableOpReadVariableOpSequential_mlp_mlp_0bias*
_output_shapes
:
*
dtype0
�
Sequential_mlp_mlp_0weightVarHandleOp*
_output_shapes
: *+

debug_nameSequential_mlp_mlp_0weight/*
dtype0*
shape:	�
*+
shared_nameSequential_mlp_mlp_0weight
�
.Sequential_mlp_mlp_0weight/Read/ReadVariableOpReadVariableOpSequential_mlp_mlp_0weight*
_output_shapes
:	�
*
dtype0
�
adam/iterationVarHandleOp*
_output_shapes
: *

debug_nameadam/iteration/*
dtype0	*
shape: *
shared_nameadam/iteration
i
"adam/iteration/Read/ReadVariableOpReadVariableOpadam/iteration*
_output_shapes
: *
dtype0	
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpadam/iteration*
_class
loc:@Variable_4*
_output_shapes
: *
dtype0	
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0	*
shape: *
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0	
a
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
: *
dtype0	
{
serving_default_inputsPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputsSequential_mlp_mlp_0weightSequential_mlp_mlp_0bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU 2J 8� �J *-
f(R&
$__inference_signature_wrapper_325722

NoOpNoOp
�

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�	
value�	B�	 B�	
�

layers
in_dims
layer_types
	optimizer
metrics
training_messg
validation_messg
__call__
	
train_step

validation_step

signatures*

0*
* 
* 
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations

_momentums
_velocities*
* 
* 
* 
)
trace_0
trace_1
trace_2* 

trace_0* 

trace_0
trace_1* 

serving_default* 

w
b*
'
0
1
2
3
4*

0
1*
* 
TN
VARIABLE_VALUE
Variable_40optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
YS
VARIABLE_VALUESequential_mlp_mlp_0weight%layers/0/w/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUESequential_mlp_mlp_0bias%layers/0/b/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_31optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_21optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUE
Variable_11optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEVariable1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename
Variable_4Sequential_mlp_mlp_0weightSequential_mlp_mlp_0bias
Variable_3
Variable_2
Variable_1VariableConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *(
f#R!
__inference__traced_save_326217
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_4Sequential_mlp_mlp_0weightSequential_mlp_mlp_0bias
Variable_3
Variable_2
Variable_1Variable*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *+
f&R$
"__inference__traced_restore_326247��
�
�
__forward___call___325942
inputs_01
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity
softmax
relu
matmul_readvariableop
truediv
sqrt
sub

inputs
moments_stopgradient��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpq
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
moments/meanMeaninputs_0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: �
moments/SquaredDifferenceSquaredDifferenceinputs_0moments/StopGradient:output:0*
T0*
_output_shapes
:	 �u
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(6
sub_0Subinputs_0moments/mean:output:0*
T0J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5`
addAddV2moments/variance:output:0add/y:output:0*
T0*
_output_shapes

: >
SqrtSqrtadd:z:0*
T0*
_output_shapes

: 2
	truediv_0RealDiv	sub_0:z:0Sqrt:y:0*
T0u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0g
MatMulMatMultruediv_0:z:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: 
O
SoftmaxSoftmaxRelu:activations:0*
T0*
_output_shapes

: 
W
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*
_output_shapes

: 
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"
inputsinputs_0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"5
moments_stopgradientmoments/StopGradient:output:0"
reluRelu:activations:0"
softmaxSoftmax:softmax:0"
sqrtSqrt:y:0"
sub	sub_0:z:0"
truedivtruediv_0:z:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	 �: : *I
backward_function_name/-__inference___backward___call___325868_32594320
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:G C

_output_shapes
:	 �
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_325722

inputs
unknown:	�

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU 2J 8� �J *$
fR
__inference___call___325712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name325718:&"
 
_user_specified_name325716:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
#__inference_internal_grad_fn_326170
result_grads_0
result_grads_1
result_grads_2
result_grads_3

identity_2

identity_3I
IdentityIdentityresult_grads_0*
T0*
_output_shapes
:
P

Identity_1Identityresult_grads_1*
T0*
_output_shapes
:	�
�
	IdentityN	IdentityNresult_grads_0result_grads_1result_grads_0result_grads_1*
T
2*,
_gradient_op_typeCustomGradient-326161*6
_output_shapes$
":
:	�
:
:	�
O

Identity_2IdentityIdentityN:output:0*
T0*
_output_shapes
:
T

Identity_3IdentityIdentityN:output:1*
T0*
_output_shapes
:	�
"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":
:	�
:
:	�
:OK

_output_shapes
:	�

(
_user_specified_nameresult_grads_3:JF

_output_shapes
:

(
_user_specified_nameresult_grads_2:OK

_output_shapes
:	�

(
_user_specified_nameresult_grads_1:J F

_output_shapes
:

(
_user_specified_nameresult_grads_0
�
�
__inference___call___325745

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpq
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
���������
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*
_output_shapes
:	 �u
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(S
subSubinputsmoments/mean:output:0*
T0*
_output_shapes
:	 �J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5`
addAddV2moments/variance:output:0add/y:output:0*
T0*
_output_shapes

: >
SqrtSqrtadd:z:0*
T0*
_output_shapes

: O
truedivRealDivsub:z:0Sqrt:y:0*
T0*
_output_shapes
:	 �u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0e
MatMulMatMultruediv:z:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: 
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: 
G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

: 
O
SoftmaxSoftmaxRelu:activations:0*
T0*
_output_shapes

: 
W
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*
_output_shapes

: 
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	 �: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:G C

_output_shapes
:	 �
 
_user_specified_nameinputs
� 
�
"__inference_validation_step_326108
x_val_batch
y_val_batch
unknown:	�

	unknown_0:

identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallx_val_batchunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU 2J 8� �J *$
fR
__inference___call___325768j
categorical_crossentropy/CastCasty_val_batch*

DstT0*

SrcT0*
_output_shapes

:
y
.categorical_crossentropy/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
categorical_crossentropy/SumSum StatefulPartitionedCall:output:07categorical_crossentropy/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
 categorical_crossentropy/truedivRealDiv StatefulPartitionedCall:output:0%categorical_crossentropy/Sum:output:0*
T0*
_output_shapes

:
u
0categorical_crossentropy/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
.categorical_crossentropy/clip_by_value/MinimumMinimum$categorical_crossentropy/truediv:z:09categorical_crossentropy/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

:
m
(categorical_crossentropy/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
&categorical_crossentropy/clip_by_valueMaximum2categorical_crossentropy/clip_by_value/Minimum:z:01categorical_crossentropy/clip_by_value/y:output:0*
T0*
_output_shapes

:
x
categorical_crossentropy/LogLog*categorical_crossentropy/clip_by_value:z:0*
T0*
_output_shapes

:
�
categorical_crossentropy/mulMul!categorical_crossentropy/Cast:y:0 categorical_crossentropy/Log:y:0*
T0*
_output_shapes

:
{
0categorical_crossentropy/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
categorical_crossentropy/Sum_1Sum categorical_crossentropy/mul:z:09categorical_crossentropy/Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:q
categorical_crossentropy/NegNeg'categorical_crossentropy/Sum_1:output:0*
T0*
_output_shapes
:h
categorical_crossentropy/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
categorical_crossentropy/Sum_2Sum categorical_crossentropy/Neg:y:0'categorical_crossentropy/Const:output:0*
T0*
_output_shapes
: h
categorical_crossentropy/ShapeConst*
_output_shapes
:*
dtype0*
valueB:j
 categorical_crossentropy/Const_1Const*
_output_shapes
:*
dtype0*
valueB:j
 categorical_crossentropy/Const_2Const*
_output_shapes
:*
dtype0*
valueB: �
categorical_crossentropy/ProdProd)categorical_crossentropy/Const_1:output:0)categorical_crossentropy/Const_2:output:0*
T0*
_output_shapes
: 
categorical_crossentropy/Cast_1Cast&categorical_crossentropy/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"categorical_crossentropy/truediv_1RealDiv'categorical_crossentropy/Sum_2:output:0#categorical_crossentropy/Cast_1:y:0*
T0*
_output_shapes
: d
IdentityIdentity&categorical_crossentropy/truediv_1:z:0^NoOp*
T0*
_output_shapes
: h

Identity_1Identity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:	�:
: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name326082:&"
 
_user_specified_name326080:KG

_output_shapes

:

%
_user_specified_namey_val_batch:L H

_output_shapes
:	�
%
_user_specified_namex_val_batch
�
�
__inference___call___325712

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpq
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������u
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(\
subSubinputsmoments/mean:output:0*
T0*(
_output_shapes
:����������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:���������G
SqrtSqrtadd:z:0*
T0*'
_output_shapes
:���������X
truedivRealDivsub:z:0Sqrt:y:0*
T0*(
_output_shapes
:����������u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0n
MatMulMatMultruediv:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
X
SoftmaxSoftmaxRelu:activations:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�U
�
-__inference___backward___call___325868_325943
placeholder&
"gradients_softmax_grad_mul_softmax%
!gradients_relu_grad_relugrad_relu6
2gradients_matmul_grad_matmul_matmul_readvariableop*
&gradients_matmul_grad_matmul_1_truediv'
#gradients_truediv_grad_realdiv_sqrt"
gradients_truediv_grad_neg_sub7
3gradients_moments_squareddifference_grad_sub_inputsE
Agradients_moments_squareddifference_grad_sub_moments_stopgradient
identity

identity_1

identity_2U
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes

: 
�
gradients/Softmax_grad/mulMulgradients/grad_ys_0:output:0"gradients_softmax_grad_mul_softmax*
T0*
_output_shapes

: 
w
,gradients/Softmax_grad/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul:z:05gradients/Softmax_grad/Sum/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
gradients/Softmax_grad/subSubgradients/grad_ys_0:output:0#gradients/Softmax_grad/Sum:output:0*
T0*
_output_shapes

: 
�
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/sub:z:0"gradients_softmax_grad_mul_softmax*
T0*
_output_shapes

: 
�
gradients/Relu_grad/ReluGradReluGrad gradients/Softmax_grad/mul_1:z:0!gradients_relu_grad_relugrad_relu*
T0*
_output_shapes

: 
�
"gradients/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients/Relu_grad/ReluGrad:backprops:0*
T0*
_output_shapes
:
�
gradients/MatMul_grad/MatMulMatMul(gradients/Relu_grad/ReluGrad:backprops:02gradients_matmul_grad_matmul_matmul_readvariableop*
T0*
_output_shapes
:	 �*
grad_a(*
transpose_b(�
gradients/MatMul_grad/MatMul_1MatMul&gradients_matmul_grad_matmul_1_truediv(gradients/Relu_grad/ReluGrad:backprops:0*
T0*
_output_shapes
:	�
*
grad_b(*
transpose_a(�
gradients/truediv_grad/RealDivRealDiv&gradients/MatMul_grad/MatMul:product:0#gradients_truediv_grad_realdiv_sqrt*
T0*
_output_shapes
:	 �k
gradients/truediv_grad/NegNeggradients_truediv_grad_neg_sub*
T0*
_output_shapes
:	 ��
 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/Neg:y:0#gradients_truediv_grad_realdiv_sqrt*
T0*
_output_shapes
:	 ��
 gradients/truediv_grad/RealDiv_2RealDiv$gradients/truediv_grad/RealDiv_1:z:0#gradients_truediv_grad_realdiv_sqrt*
T0*
_output_shapes
:	 ��
gradients/truediv_grad/mulMul&gradients/MatMul_grad/MatMul:product:0$gradients/truediv_grad/RealDiv_2:z:0*
T0*
_output_shapes
:	 �m
gradients/truediv_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      o
gradients/truediv_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"       v
,gradients/truediv_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
gradients/truediv_grad/SumSumgradients/truediv_grad/mul:z:05gradients/truediv_grad/Sum/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
gradients/truediv_grad/ReshapeReshape#gradients/truediv_grad/Sum:output:0'gradients/truediv_grad/Shape_1:output:0*
T0*
_output_shapes

: k
gradients/sub_grad/NegNeg"gradients/truediv_grad/RealDiv:z:0*
T0*
_output_shapes
:	 �i
gradients/sub_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      k
gradients/sub_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"       r
(gradients/sub_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
gradients/sub_grad/SumSumgradients/sub_grad/Neg:y:01gradients/sub_grad/Sum/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sum:output:0#gradients/sub_grad/Shape_1:output:0*
T0*
_output_shapes

: �
gradients/Sqrt_grad/SqrtGradSqrtGrad#gradients_truediv_grad_realdiv_sqrt'gradients/truediv_grad/Reshape:output:0*
T0*
_output_shapes

: i
gradients/add_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       ]
gradients/add_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB r
(gradients/add_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
gradients/add_grad/SumSum gradients/Sqrt_grad/SqrtGrad:z:01gradients/add_grad/Sum/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sum:output:0#gradients/add_grad/Shape_1:output:0*
T0*
_output_shapes
: z
)gradients/moments/variance_grad/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB"       k
)gradients/moments/variance_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :�
'gradients/moments/variance_grad/MaximumMaximum2gradients/moments/variance_grad/Maximum/x:output:02gradients/moments/variance_grad/Maximum/y:output:0*
T0*
_output_shapes
:{
*gradients/moments/variance_grad/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB"      �
(gradients/moments/variance_grad/floordivFloorDiv3gradients/moments/variance_grad/floordiv/x:output:0+gradients/moments/variance_grad/Maximum:z:0*
T0*
_output_shapes
:~
-gradients/moments/variance_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
'gradients/moments/variance_grad/ReshapeReshape gradients/Sqrt_grad/SqrtGrad:z:06gradients/moments/variance_grad/Reshape/shape:output:0*
T0*
_output_shapes

: 
.gradients/moments/variance_grad/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"     �
$gradients/moments/variance_grad/TileTile0gradients/moments/variance_grad/Reshape:output:07gradients/moments/variance_grad/Tile/multiples:output:0*
T0*
_output_shapes
:	 �j
%gradients/moments/variance_grad/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  DD�
'gradients/moments/variance_grad/truedivRealDiv-gradients/moments/variance_grad/Tile:output:0.gradients/moments/variance_grad/Const:output:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes
:	 ��
/gradients/moments/SquaredDifference_grad/scalarConst(^gradients/moments/variance_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @�
,gradients/moments/SquaredDifference_grad/MulMul8gradients/moments/SquaredDifference_grad/scalar:output:0+gradients/moments/variance_grad/truediv:z:0*
T0*
_output_shapes
:	 ��
,gradients/moments/SquaredDifference_grad/subSub3gradients_moments_squareddifference_grad_sub_inputsAgradients_moments_squareddifference_grad_sub_moments_stopgradient(^gradients/moments/variance_grad/truediv*
T0*
_output_shapes
:	 ��
.gradients/moments/SquaredDifference_grad/mul_1Mul0gradients/moments/SquaredDifference_grad/Mul:z:00gradients/moments/SquaredDifference_grad/sub:z:0*
T0*
_output_shapes
:	 ��
,gradients/moments/SquaredDifference_grad/NegNeg2gradients/moments/SquaredDifference_grad/mul_1:z:0*
T0*
_output_shapes
:	 �
.gradients/moments/SquaredDifference_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      �
0gradients/moments/SquaredDifference_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"       �
>gradients/moments/SquaredDifference_grad/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
,gradients/moments/SquaredDifference_grad/SumSum0gradients/moments/SquaredDifference_grad/Neg:y:0Ggradients/moments/SquaredDifference_grad/Sum/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
0gradients/moments/SquaredDifference_grad/ReshapeReshape5gradients/moments/SquaredDifference_grad/Sum:output:09gradients/moments/SquaredDifference_grad/Shape_1:output:0*
T0*
_output_shapes

: z
)gradients/moments/mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
#gradients/moments/mean_grad/ReshapeReshape#gradients/sub_grad/Reshape:output:02gradients/moments/mean_grad/Reshape/shape:output:0*
T0*
_output_shapes

: {
*gradients/moments/mean_grad/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"     �
 gradients/moments/mean_grad/TileTile,gradients/moments/mean_grad/Reshape:output:03gradients/moments/mean_grad/Tile/multiples:output:0*
T0*
_output_shapes
:	 �f
!gradients/moments/mean_grad/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  DD�
#gradients/moments/mean_grad/truedivRealDiv)gradients/moments/mean_grad/Tile:output:0*gradients/moments/mean_grad/Const:output:0*
T0*
_output_shapes
:	 ��
gradients/AddNAddN"gradients/truediv_grad/RealDiv:z:02gradients/moments/SquaredDifference_grad/mul_1:z:0'gradients/moments/mean_grad/truediv:z:0*
N*
T0*1
_class'
%#loc:@gradients/truediv_grad/RealDiv*
_output_shapes
:	 �T
IdentityIdentitygradients/AddN:sum:0*
T0*
_output_shapes
:	 �j

Identity_1Identity(gradients/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes
:	�
h

Identity_2Identity+gradients/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:
"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*q
_input_shapes`
^: 
: 
: 
:	�
:	 �: :	 �:	 �: *4
forward_function_name__forward___call___325942:TP

_output_shapes

: 
.
_user_specified_namemoments/StopGradient:GC

_output_shapes
:	 �
 
_user_specified_nameinputs:D@

_output_shapes
:	 �

_user_specified_namesub:D@

_output_shapes

: 

_user_specified_nameSqrt:HD

_output_shapes
:	 �
!
_user_specified_name	truediv:VR

_output_shapes
:	�

/
_user_specified_nameMatMul/ReadVariableOp:D@

_output_shapes

: 


_user_specified_nameRelu:GC

_output_shapes

: 

!
_user_specified_name	Softmax:$  

_output_shapes

: 

�
�
__inference___call___325768

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpq
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
���������
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*
_output_shapes
:	�u
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(S
subSubinputsmoments/mean:output:0*
T0*
_output_shapes
:	�J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5`
addAddV2moments/variance:output:0add/y:output:0*
T0*
_output_shapes

:>
SqrtSqrtadd:z:0*
T0*
_output_shapes

:O
truedivRealDivsub:z:0Sqrt:y:0*
T0*
_output_shapes
:	�u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0e
MatMulMatMultruediv:z:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:
G
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:
O
SoftmaxSoftmaxRelu:activations:0*
T0*
_output_shapes

:
W
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*
_output_shapes

:
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�
�
__inference___call___325791

inputs1
matmul_readvariableop_resource:	�
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpq
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:����������
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������u
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
����������
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:���������*
	keep_dims(\
subSubinputsmoments/mean:output:0*
T0*(
_output_shapes
:����������J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�5i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:���������G
SqrtSqrtadd:z:0*
T0*'
_output_shapes
:���������X
truedivRealDivsub:z:0Sqrt:y:0*
T0*(
_output_shapes
:����������u
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype0n
MatMulMatMultruediv:z:0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
X
SoftmaxSoftmaxRelu:activations:0*
T0*'
_output_shapes
:���������
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
"__inference_validation_step_326076
x_val_batch
y_val_batch
unknown:	�

	unknown_0:

identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallx_val_batchunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

: 
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU 2J 8� �J *$
fR
__inference___call___325745j
categorical_crossentropy/CastCasty_val_batch*

DstT0*

SrcT0*
_output_shapes

: 
y
.categorical_crossentropy/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
categorical_crossentropy/SumSum StatefulPartitionedCall:output:07categorical_crossentropy/Sum/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
 categorical_crossentropy/truedivRealDiv StatefulPartitionedCall:output:0%categorical_crossentropy/Sum:output:0*
T0*
_output_shapes

: 
u
0categorical_crossentropy/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
.categorical_crossentropy/clip_by_value/MinimumMinimum$categorical_crossentropy/truediv:z:09categorical_crossentropy/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

: 
m
(categorical_crossentropy/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
&categorical_crossentropy/clip_by_valueMaximum2categorical_crossentropy/clip_by_value/Minimum:z:01categorical_crossentropy/clip_by_value/y:output:0*
T0*
_output_shapes

: 
x
categorical_crossentropy/LogLog*categorical_crossentropy/clip_by_value:z:0*
T0*
_output_shapes

: 
�
categorical_crossentropy/mulMul!categorical_crossentropy/Cast:y:0 categorical_crossentropy/Log:y:0*
T0*
_output_shapes

: 
{
0categorical_crossentropy/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
categorical_crossentropy/Sum_1Sum categorical_crossentropy/mul:z:09categorical_crossentropy/Sum_1/reduction_indices:output:0*
T0*
_output_shapes
: q
categorical_crossentropy/NegNeg'categorical_crossentropy/Sum_1:output:0*
T0*
_output_shapes
: h
categorical_crossentropy/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
categorical_crossentropy/Sum_2Sum categorical_crossentropy/Neg:y:0'categorical_crossentropy/Const:output:0*
T0*
_output_shapes
: h
categorical_crossentropy/ShapeConst*
_output_shapes
:*
dtype0*
valueB: j
 categorical_crossentropy/Const_1Const*
_output_shapes
:*
dtype0*
valueB: j
 categorical_crossentropy/Const_2Const*
_output_shapes
:*
dtype0*
valueB: �
categorical_crossentropy/ProdProd)categorical_crossentropy/Const_1:output:0)categorical_crossentropy/Const_2:output:0*
T0*
_output_shapes
: 
categorical_crossentropy/Cast_1Cast&categorical_crossentropy/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"categorical_crossentropy/truediv_1RealDiv'categorical_crossentropy/Sum_2:output:0#categorical_crossentropy/Cast_1:y:0*
T0*
_output_shapes
: d
IdentityIdentity&categorical_crossentropy/truediv_1:z:0^NoOp*
T0*
_output_shapes
: h

Identity_1Identity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: 
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:	 �: 
: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name326050:&"
 
_user_specified_name326048:KG

_output_shapes

: 

%
_user_specified_namey_val_batch:L H

_output_shapes
:	 �
%
_user_specified_namex_val_batch
��
�
__inference_train_step_326044
x_batch
y_batch
unknown:	�

	unknown_0:
>
4adam_exponentialdecay_cast_2_readvariableop_resource:	 0
"adam_sub_2_readvariableop_resource:
0
"adam_sub_3_readvariableop_resource:
5
"adam_sub_6_readvariableop_resource:	�
5
"adam_sub_7_readvariableop_resource:	�

identity

identity_1��StatefulPartitionedCall�adam/Add/ReadVariableOp�adam/Add_2/ReadVariableOp�adam/Add_4/ReadVariableOp�adam/AssignAddVariableOp�adam/AssignAddVariableOp_1�adam/AssignAddVariableOp_2�adam/AssignAddVariableOp_3�adam/AssignSubVariableOp�adam/AssignSubVariableOp_1�adam/AssignVariableOp�adam/Cast_3/ReadVariableOp�adam/Cast_7/ReadVariableOp�+adam/ExponentialDecay/Cast_2/ReadVariableOp�adam/Mul_3/ReadVariableOp�adam/Mul_7/ReadVariableOp�adam/Sub_2/ReadVariableOp�adam/Sub_3/ReadVariableOp�adam/Sub_6/ReadVariableOp�adam/Sub_7/ReadVariableOp�
StatefulPartitionedCallStatefulPartitionedCallx_batchunknown	unknown_0*
Tin
2*
Tout
2	*
_collective_manager_ids
 *r
_output_shapes`
^: 
: 
: 
:	�
:	 �: :	 �:	 �: *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU 2J 8� �J *"
fR
__forward___call___325942f
categorical_crossentropy/CastCasty_batch*

DstT0*

SrcT0*
_output_shapes

: 
y
.categorical_crossentropy/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
categorical_crossentropy/SumSum StatefulPartitionedCall:output:07categorical_crossentropy/Sum/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
 categorical_crossentropy/truedivRealDiv StatefulPartitionedCall:output:0%categorical_crossentropy/Sum:output:0*
T0*
_output_shapes

: 
u
0categorical_crossentropy/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��?�
.categorical_crossentropy/clip_by_value/MinimumMinimum$categorical_crossentropy/truediv:z:09categorical_crossentropy/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

: 
m
(categorical_crossentropy/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *���3�
&categorical_crossentropy/clip_by_valueMaximum2categorical_crossentropy/clip_by_value/Minimum:z:01categorical_crossentropy/clip_by_value/y:output:0*
T0*
_output_shapes

: 
x
categorical_crossentropy/LogLog*categorical_crossentropy/clip_by_value:z:0*
T0*
_output_shapes

: 
�
categorical_crossentropy/mulMul!categorical_crossentropy/Cast:y:0 categorical_crossentropy/Log:y:0*
T0*
_output_shapes

: 
{
0categorical_crossentropy/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
categorical_crossentropy/Sum_1Sum categorical_crossentropy/mul:z:09categorical_crossentropy/Sum_1/reduction_indices:output:0*
T0*
_output_shapes
: q
categorical_crossentropy/NegNeg'categorical_crossentropy/Sum_1:output:0*
T0*
_output_shapes
: h
categorical_crossentropy/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
categorical_crossentropy/Sum_2Sum categorical_crossentropy/Neg:y:0'categorical_crossentropy/Const:output:0*
T0*
_output_shapes
: h
categorical_crossentropy/ShapeConst*
_output_shapes
:*
dtype0*
valueB: j
 categorical_crossentropy/Const_1Const*
_output_shapes
:*
dtype0*
valueB: j
 categorical_crossentropy/Const_2Const*
_output_shapes
:*
dtype0*
valueB: �
categorical_crossentropy/ProdProd)categorical_crossentropy/Const_1:output:0)categorical_crossentropy/Const_2:output:0*
T0*
_output_shapes
: 
categorical_crossentropy/Cast_1Cast&categorical_crossentropy/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: �
"categorical_crossentropy/truediv_1RealDiv'categorical_crossentropy/Sum_2:output:0#categorical_crossentropy/Cast_1:y:0*
T0*
_output_shapes
: I
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
8gradient_tape/categorical_crossentropy/truediv_1/RealDivRealDivones:output:0#categorical_crossentropy/Cast_1:y:0*
T0*
_output_shapes
: �
4gradient_tape/categorical_crossentropy/truediv_1/NegNeg'categorical_crossentropy/Sum_2:output:0*
T0*
_output_shapes
: �
:gradient_tape/categorical_crossentropy/truediv_1/RealDiv_1RealDiv8gradient_tape/categorical_crossentropy/truediv_1/Neg:y:0#categorical_crossentropy/Cast_1:y:0*
T0*
_output_shapes
: �
:gradient_tape/categorical_crossentropy/truediv_1/RealDiv_2RealDiv>gradient_tape/categorical_crossentropy/truediv_1/RealDiv_1:z:0#categorical_crossentropy/Cast_1:y:0*
T0*
_output_shapes
: �
4gradient_tape/categorical_crossentropy/truediv_1/mulMulones:output:0>gradient_tape/categorical_crossentropy/truediv_1/RealDiv_2:z:0*
T0*
_output_shapes
: y
6gradient_tape/categorical_crossentropy/truediv_1/ShapeConst*
_output_shapes
: *
dtype0*
valueB {
8gradient_tape/categorical_crossentropy/truediv_1/Shape_1Const*
_output_shapes
: *
dtype0*
valueB ~
4gradient_tape/categorical_crossentropy/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
.gradient_tape/categorical_crossentropy/ReshapeReshape<gradient_tape/categorical_crossentropy/truediv_1/RealDiv:z:0=gradient_tape/categorical_crossentropy/Reshape/shape:output:0*
T0*
_output_shapes
:v
,gradient_tape/categorical_crossentropy/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
+gradient_tape/categorical_crossentropy/TileTile7gradient_tape/categorical_crossentropy/Reshape:output:05gradient_tape/categorical_crossentropy/Const:output:0*
T0*
_output_shapes
: �
*gradient_tape/categorical_crossentropy/NegNeg4gradient_tape/categorical_crossentropy/Tile:output:0*
T0*
_output_shapes
: �
0gradient_tape/categorical_crossentropy/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB"       r
0gradient_tape/categorical_crossentropy/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :�
.gradient_tape/categorical_crossentropy/MaximumMaximum9gradient_tape/categorical_crossentropy/Maximum/x:output:09gradient_tape/categorical_crossentropy/Maximum/y:output:0*
T0*
_output_shapes
:�
1gradient_tape/categorical_crossentropy/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB"    
   �
/gradient_tape/categorical_crossentropy/floordivFloorDiv:gradient_tape/categorical_crossentropy/floordiv/x:output:02gradient_tape/categorical_crossentropy/Maximum:z:0*
T0*
_output_shapes
:�
6gradient_tape/categorical_crossentropy/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
0gradient_tape/categorical_crossentropy/Reshape_1Reshape.gradient_tape/categorical_crossentropy/Neg:y:0?gradient_tape/categorical_crossentropy/Reshape_1/shape:output:0*
T0*
_output_shapes

: �
7gradient_tape/categorical_crossentropy/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"   
   �
-gradient_tape/categorical_crossentropy/Tile_1Tile9gradient_tape/categorical_crossentropy/Reshape_1:output:0@gradient_tape/categorical_crossentropy/Tile_1/multiples:output:0*
T0*
_output_shapes

: 
�
.gradient_tape/categorical_crossentropy/mul/MulMul6gradient_tape/categorical_crossentropy/Tile_1:output:0 categorical_crossentropy/Log:y:0*
T0*
_output_shapes

: 
�
0gradient_tape/categorical_crossentropy/mul/Mul_1Mul6gradient_tape/categorical_crossentropy/Tile_1:output:0!categorical_crossentropy/Cast:y:0*
T0*&
 _has_manual_control_dependencies(*
_output_shapes

: 
�
1gradient_tape/categorical_crossentropy/Reciprocal
Reciprocal*categorical_crossentropy/clip_by_value:z:01^gradient_tape/categorical_crossentropy/mul/Mul_1*
T0*
_output_shapes

: 
�
*gradient_tape/categorical_crossentropy/mulMul4gradient_tape/categorical_crossentropy/mul/Mul_1:z:05gradient_tape/categorical_crossentropy/Reciprocal:y:0*
T0*
_output_shapes

: 
�
?gradient_tape/categorical_crossentropy/clip_by_value/zeros_likeConst*
_output_shapes

: 
*
dtype0*
valueB 
*    �
Agradient_tape/categorical_crossentropy/clip_by_value/GreaterEqualGreaterEqual2categorical_crossentropy/clip_by_value/Minimum:z:01categorical_crossentropy/clip_by_value/y:output:0*
T0*
_output_shapes

: 
�
=gradient_tape/categorical_crossentropy/clip_by_value/SelectV2SelectV2Egradient_tape/categorical_crossentropy/clip_by_value/GreaterEqual:z:0.gradient_tape/categorical_crossentropy/mul:z:0Hgradient_tape/categorical_crossentropy/clip_by_value/zeros_like:output:0*
T0*
_output_shapes

: 
�
Agradient_tape/categorical_crossentropy/clip_by_value/zeros_like_1Const*
_output_shapes

: 
*
dtype0*
valueB 
*    �
>gradient_tape/categorical_crossentropy/clip_by_value/LessEqual	LessEqual$categorical_crossentropy/truediv:z:09categorical_crossentropy/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes

: 
�
?gradient_tape/categorical_crossentropy/clip_by_value/SelectV2_1SelectV2Bgradient_tape/categorical_crossentropy/clip_by_value/LessEqual:z:0Fgradient_tape/categorical_crossentropy/clip_by_value/SelectV2:output:0Jgradient_tape/categorical_crossentropy/clip_by_value/zeros_like_1:output:0*
T0*
_output_shapes

: 
�
6gradient_tape/categorical_crossentropy/truediv/RealDivRealDivHgradient_tape/categorical_crossentropy/clip_by_value/SelectV2_1:output:0%categorical_crossentropy/Sum:output:0*
T0*
_output_shapes

: 
�
2gradient_tape/categorical_crossentropy/truediv/NegNeg StatefulPartitionedCall:output:0*
T0*
_output_shapes

: 
�
8gradient_tape/categorical_crossentropy/truediv/RealDiv_1RealDiv6gradient_tape/categorical_crossentropy/truediv/Neg:y:0%categorical_crossentropy/Sum:output:0*
T0*
_output_shapes

: 
�
8gradient_tape/categorical_crossentropy/truediv/RealDiv_2RealDiv<gradient_tape/categorical_crossentropy/truediv/RealDiv_1:z:0%categorical_crossentropy/Sum:output:0*
T0*
_output_shapes

: 
�
2gradient_tape/categorical_crossentropy/truediv/mulMulHgradient_tape/categorical_crossentropy/clip_by_value/SelectV2_1:output:0<gradient_tape/categorical_crossentropy/truediv/RealDiv_2:z:0*
T0*
_output_shapes

: 
�
4gradient_tape/categorical_crossentropy/truediv/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   �
6gradient_tape/categorical_crossentropy/truediv/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"       �
Dgradient_tape/categorical_crossentropy/truediv/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
2gradient_tape/categorical_crossentropy/truediv/SumSum6gradient_tape/categorical_crossentropy/truediv/mul:z:0Mgradient_tape/categorical_crossentropy/truediv/Sum/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(�
6gradient_tape/categorical_crossentropy/truediv/ReshapeReshape;gradient_tape/categorical_crossentropy/truediv/Sum:output:0?gradient_tape/categorical_crossentropy/truediv/Shape_1:output:0*
T0*
_output_shapes

: �
6gradient_tape/categorical_crossentropy/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
0gradient_tape/categorical_crossentropy/Reshape_2Reshape?gradient_tape/categorical_crossentropy/truediv/Reshape:output:0?gradient_tape/categorical_crossentropy/Reshape_2/shape:output:0*
T0*
_output_shapes

: �
7gradient_tape/categorical_crossentropy/Tile_2/multiplesConst*
_output_shapes
:*
dtype0*
valueB"   
   �
-gradient_tape/categorical_crossentropy/Tile_2Tile9gradient_tape/categorical_crossentropy/Reshape_2:output:0@gradient_tape/categorical_crossentropy/Tile_2/multiples:output:0*
T0*
_output_shapes

: 
�
AddNAddN:gradient_tape/categorical_crossentropy/truediv/RealDiv:z:06gradient_tape/categorical_crossentropy/Tile_2:output:0*
N*
T0*
_output_shapes

: 
�
PartitionedCallPartitionedCall
AddN:sum:0 StatefulPartitionedCall:output:1 StatefulPartitionedCall:output:2 StatefulPartitionedCall:output:3 StatefulPartitionedCall:output:4 StatefulPartitionedCall:output:5 StatefulPartitionedCall:output:6 StatefulPartitionedCall:output:7 StatefulPartitionedCall:output:8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:	 �:	�
:
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *6
f1R/
-__inference___backward___call___325868_325943`
adam/ExponentialDecay/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *o�:_
adam/ExponentialDecay/Cast/xConst*
_output_shapes
: *
dtype0*
value
B :�y
adam/ExponentialDecay/CastCast%adam/ExponentialDecay/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: c
adam/ExponentialDecay/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *{n?�
+adam/ExponentialDecay/Cast_2/ReadVariableOpReadVariableOp4adam_exponentialdecay_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0	�
adam/ExponentialDecay/Cast_2Cast3adam/ExponentialDecay/Cast_2/ReadVariableOp:value:0*

DstT0*

SrcT0	*
_output_shapes
: �
adam/ExponentialDecay/truedivRealDiv adam/ExponentialDecay/Cast_2:y:0adam/ExponentialDecay/Cast:y:0*
T0*
_output_shapes
: h
adam/ExponentialDecay/FloorFloor!adam/ExponentialDecay/truediv:z:0*
T0*
_output_shapes
: �
adam/ExponentialDecay/PowPow'adam/ExponentialDecay/Cast_1/x:output:0adam/ExponentialDecay/Floor:y:0*
T0*
_output_shapes
: �
adam/ExponentialDecay/MulMul$adam/ExponentialDecay/Const:output:0adam/ExponentialDecay/Pow:z:0*
T0*
_output_shapes
: X
adam/IdentityIdentityPartitionedCall:output:2*
T0*
_output_shapes
:
_
adam/Identity_1IdentityPartitionedCall:output:1*
T0*
_output_shapes
:	�
�
adam/IdentityN	IdentityNPartitionedCall:output:2PartitionedCall:output:1PartitionedCall:output:2PartitionedCall:output:1*
T
2*,
_gradient_op_typeCustomGradient-325966*6
_output_shapes$
":
:	�
:
:	�
L

adam/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R�
adam/Add/ReadVariableOpReadVariableOp4adam_exponentialdecay_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0	h
adam/AddAddV2adam/Add/ReadVariableOp:value:0adam/Const:output:0*
T0	*
_output_shapes
: O
	adam/CastCastadam/Add:z:0*

DstT0*

SrcT0	*
_output_shapes
: R
adam/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?W
adam/PowPowadam/Cast_1/x:output:0adam/Cast:y:0*
T0*
_output_shapes
: R
adam/Cast_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?Y

adam/Pow_1Powadam/Cast_2/x:output:0adam/Cast:y:0*
T0*
_output_shapes
: O

adam/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
adam/subSubadam/sub/x:output:0adam/Pow_1:z:0*
T0*
_output_shapes
: @
	adam/SqrtSqrtadam/sub:z:0*
T0*
_output_shapes
: ^
adam/mulMuladam/ExponentialDecay/Mul:z:0adam/Sqrt:y:0*
T0*
_output_shapes
: Q
adam/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?W

adam/sub_1Subadam/sub_1/x:output:0adam/Pow:z:0*
T0*
_output_shapes
: V
adam/truedivRealDivadam/mul:z:0adam/sub_1:z:0*
T0*
_output_shapes
: x
adam/Sub_2/ReadVariableOpReadVariableOp"adam_sub_2_readvariableop_resource*
_output_shapes
:
*
dtype0r

adam/Sub_2Subadam/IdentityN:output:0!adam/Sub_2/ReadVariableOp:value:0*
T0*
_output_shapes
:
Q
adam/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *���=]

adam/Mul_1Muladam/Sub_2:z:0adam/Const_1:output:0*
T0*
_output_shapes
:
�
adam/AssignAddVariableOpAssignAddVariableOp"adam_sub_2_readvariableop_resourceadam/Mul_1:z:0^adam/Sub_2/ReadVariableOp*
_output_shapes
 *
dtype0S
adam/SquareSquareadam/IdentityN:output:0*
T0*
_output_shapes
:
x
adam/Sub_3/ReadVariableOpReadVariableOp"adam_sub_3_readvariableop_resource*
_output_shapes
:
*
dtype0j

adam/Sub_3Subadam/Square:y:0!adam/Sub_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
Q
adam/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *o�:]

adam/Mul_2Muladam/Sub_3:z:0adam/Const_2:output:0*
T0*
_output_shapes
:
�
adam/AssignAddVariableOp_1AssignAddVariableOp"adam_sub_3_readvariableop_resourceadam/Mul_2:z:0^adam/Sub_3/ReadVariableOp*
_output_shapes
 *
dtype0�
adam/Mul_3/ReadVariableOpReadVariableOp"adam_sub_2_readvariableop_resource^adam/AssignAddVariableOp*
_output_shapes
:
*
dtype0k

adam/Mul_3Mul!adam/Mul_3/ReadVariableOp:value:0adam/truediv:z:0*
T0*
_output_shapes
:
�
adam/Cast_3/ReadVariableOpReadVariableOp"adam_sub_3_readvariableop_resource^adam/AssignAddVariableOp_1*
_output_shapes
:
*
dtype0\
adam/Sqrt_1Sqrt"adam/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
Q
adam/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *���3`

adam/Add_1AddV2adam/Sqrt_1:y:0adam/Const_3:output:0*
T0*
_output_shapes
:
^
adam/truediv_1RealDivadam/Mul_3:z:0adam/Add_1:z:0*
T0*
_output_shapes
:
�
adam/AssignSubVariableOpAssignSubVariableOp	unknown_0adam/truediv_1:z:0^StatefulPartitionedCall*
_output_shapes
 *
dtype0N
adam/Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R�
adam/Add_2/ReadVariableOpReadVariableOp4adam_exponentialdecay_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0	n

adam/Add_2AddV2!adam/Add_2/ReadVariableOp:value:0adam/Const_4:output:0*
T0	*
_output_shapes
: S
adam/Cast_4Castadam/Add_2:z:0*

DstT0*

SrcT0	*
_output_shapes
: R
adam/Cast_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *fff?[

adam/Pow_2Powadam/Cast_5/x:output:0adam/Cast_4:y:0*
T0*
_output_shapes
: R
adam/Cast_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *w�?[

adam/Pow_3Powadam/Cast_6/x:output:0adam/Cast_4:y:0*
T0*
_output_shapes
: Q
adam/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

adam/sub_4Subadam/sub_4/x:output:0adam/Pow_3:z:0*
T0*
_output_shapes
: D
adam/Sqrt_2Sqrtadam/sub_4:z:0*
T0*
_output_shapes
: b

adam/mul_4Muladam/ExponentialDecay/Mul:z:0adam/Sqrt_2:y:0*
T0*
_output_shapes
: Q
adam/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?Y

adam/sub_5Subadam/sub_5/x:output:0adam/Pow_2:z:0*
T0*
_output_shapes
: Z
adam/truediv_2RealDivadam/mul_4:z:0adam/sub_5:z:0*
T0*
_output_shapes
: }
adam/Sub_6/ReadVariableOpReadVariableOp"adam_sub_6_readvariableop_resource*
_output_shapes
:	�
*
dtype0w

adam/Sub_6Subadam/IdentityN:output:1!adam/Sub_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�
Q
adam/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *���=b

adam/Mul_5Muladam/Sub_6:z:0adam/Const_5:output:0*
T0*
_output_shapes
:	�
�
adam/AssignAddVariableOp_2AssignAddVariableOp"adam_sub_6_readvariableop_resourceadam/Mul_5:z:0^adam/Sub_6/ReadVariableOp*
_output_shapes
 *
dtype0Z
adam/Square_1Squareadam/IdentityN:output:1*
T0*
_output_shapes
:	�
}
adam/Sub_7/ReadVariableOpReadVariableOp"adam_sub_7_readvariableop_resource*
_output_shapes
:	�
*
dtype0q

adam/Sub_7Subadam/Square_1:y:0!adam/Sub_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�
Q
adam/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *o�:b

adam/Mul_6Muladam/Sub_7:z:0adam/Const_6:output:0*
T0*
_output_shapes
:	�
�
adam/AssignAddVariableOp_3AssignAddVariableOp"adam_sub_7_readvariableop_resourceadam/Mul_6:z:0^adam/Sub_7/ReadVariableOp*
_output_shapes
 *
dtype0�
adam/Mul_7/ReadVariableOpReadVariableOp"adam_sub_6_readvariableop_resource^adam/AssignAddVariableOp_2*
_output_shapes
:	�
*
dtype0r

adam/Mul_7Mul!adam/Mul_7/ReadVariableOp:value:0adam/truediv_2:z:0*
T0*
_output_shapes
:	�
�
adam/Cast_7/ReadVariableOpReadVariableOp"adam_sub_7_readvariableop_resource^adam/AssignAddVariableOp_3*
_output_shapes
:	�
*
dtype0a
adam/Sqrt_3Sqrt"adam/Cast_7/ReadVariableOp:value:0*
T0*
_output_shapes
:	�
Q
adam/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *���3e

adam/Add_3AddV2adam/Sqrt_3:y:0adam/Const_7:output:0*
T0*
_output_shapes
:	�
c
adam/truediv_3RealDivadam/Mul_7:z:0adam/Add_3:z:0*
T0*
_output_shapes
:	�
�
adam/AssignSubVariableOp_1AssignSubVariableOpunknownadam/truediv_3:z:0^StatefulPartitionedCall*
_output_shapes
 *
dtype0N
adam/Const_8Const*
_output_shapes
: *
dtype0	*
value	B	 R�
adam/Add_4/ReadVariableOpReadVariableOp4adam_exponentialdecay_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0	n

adam/Add_4AddV2!adam/Add_4/ReadVariableOp:value:0adam/Const_8:output:0*
T0	*
_output_shapes
: �
adam/AssignVariableOpAssignVariableOp4adam_exponentialdecay_cast_2_readvariableop_resourceadam/Add_4:z:0^adam/Add/ReadVariableOp^adam/Add_2/ReadVariableOp^adam/Add_4/ReadVariableOp,^adam/ExponentialDecay/Cast_2/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(d
IdentityIdentity&categorical_crossentropy/truediv_1:z:0^NoOp*
T0*
_output_shapes
: h

Identity_1Identity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

: 
�
NoOpNoOp^StatefulPartitionedCall^adam/Add/ReadVariableOp^adam/Add_2/ReadVariableOp^adam/Add_4/ReadVariableOp^adam/AssignAddVariableOp^adam/AssignAddVariableOp_1^adam/AssignAddVariableOp_2^adam/AssignAddVariableOp_3^adam/AssignSubVariableOp^adam/AssignSubVariableOp_1^adam/AssignVariableOp^adam/Cast_3/ReadVariableOp^adam/Cast_7/ReadVariableOp,^adam/ExponentialDecay/Cast_2/ReadVariableOp^adam/Mul_3/ReadVariableOp^adam/Mul_7/ReadVariableOp^adam/Sub_2/ReadVariableOp^adam/Sub_3/ReadVariableOp^adam/Sub_6/ReadVariableOp^adam/Sub_7/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:	 �: 
: : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall22
adam/Add/ReadVariableOpadam/Add/ReadVariableOp26
adam/Add_2/ReadVariableOpadam/Add_2/ReadVariableOp26
adam/Add_4/ReadVariableOpadam/Add_4/ReadVariableOp28
adam/AssignAddVariableOp_1adam/AssignAddVariableOp_128
adam/AssignAddVariableOp_2adam/AssignAddVariableOp_228
adam/AssignAddVariableOp_3adam/AssignAddVariableOp_324
adam/AssignAddVariableOpadam/AssignAddVariableOp28
adam/AssignSubVariableOp_1adam/AssignSubVariableOp_124
adam/AssignSubVariableOpadam/AssignSubVariableOp2.
adam/AssignVariableOpadam/AssignVariableOp28
adam/Cast_3/ReadVariableOpadam/Cast_3/ReadVariableOp28
adam/Cast_7/ReadVariableOpadam/Cast_7/ReadVariableOp2Z
+adam/ExponentialDecay/Cast_2/ReadVariableOp+adam/ExponentialDecay/Cast_2/ReadVariableOp26
adam/Mul_3/ReadVariableOpadam/Mul_3/ReadVariableOp26
adam/Mul_7/ReadVariableOpadam/Mul_7/ReadVariableOp26
adam/Sub_2/ReadVariableOpadam/Sub_2/ReadVariableOp26
adam/Sub_3/ReadVariableOpadam/Sub_3/ReadVariableOp26
adam/Sub_6/ReadVariableOpadam/Sub_6/ReadVariableOp26
adam/Sub_7/ReadVariableOpadam/Sub_7/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:&"
 
_user_specified_name325797:&"
 
_user_specified_name325795:GC

_output_shapes

: 

!
_user_specified_name	y_batch:H D

_output_shapes
:	 �
!
_user_specified_name	x_batch
�?
�
__inference__traced_save_326217
file_prefix+
!read_disablecopyonread_variable_4:	 F
3read_1_disablecopyonread_sequential_mlp_mlp_0weight:	�
?
1read_2_disablecopyonread_sequential_mlp_mlp_0bias:
1
#read_3_disablecopyonread_variable_3:
1
#read_4_disablecopyonread_variable_2:
6
#read_5_disablecopyonread_variable_1:	�
4
!read_6_disablecopyonread_variable:	�

savev2_const
identity_15��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: d
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_4*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_4^Read/DisableCopyOnRead*
_output_shapes
: *
dtype0	R
IdentityIdentityRead/ReadVariableOp:value:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: x
Read_1/DisableCopyOnReadDisableCopyOnRead3read_1_disablecopyonread_sequential_mlp_mlp_0weight*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp3read_1_disablecopyonread_sequential_mlp_mlp_0weight^Read_1/DisableCopyOnRead*
_output_shapes
:	�
*
dtype0_

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�
d

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
v
Read_2/DisableCopyOnReadDisableCopyOnRead1read_2_disablecopyonread_sequential_mlp_mlp_0bias*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp1read_2_disablecopyonread_sequential_mlp_mlp_0bias^Read_2/DisableCopyOnRead*
_output_shapes
:
*
dtype0Z

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
:
_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:
h
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_variable_3*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_variable_3^Read_3/DisableCopyOnRead*
_output_shapes
:
*
dtype0Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:
_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:
h
Read_4/DisableCopyOnReadDisableCopyOnRead#read_4_disablecopyonread_variable_2*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp#read_4_disablecopyonread_variable_2^Read_4/DisableCopyOnRead*
_output_shapes
:
*
dtype0Z

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
:
_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:
h
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_variable_1*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_variable_1^Read_5/DisableCopyOnRead*
_output_shapes
:	�
*
dtype0`
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
:	�
f
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
f
Read_6/DisableCopyOnReadDisableCopyOnRead!read_6_disablecopyonread_variable*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp!read_6_disablecopyonread_variable^Read_6/DisableCopyOnRead*
_output_shapes
:	�
*
dtype0`
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*
_output_shapes
:	�
f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	�
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB%layers/0/w/.ATTRIBUTES/VARIABLE_VALUEB%layers/0/b/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes

2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_14Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_15IdentityIdentity_14:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:84
2
_user_specified_nameSequential_mlp_mlp_0bias::6
4
_user_specified_nameSequential_mlp_mlp_0weight:*&
$
_user_specified_name
Variable_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�%
�
"__inference__traced_restore_326247
file_prefix%
assignvariableop_variable_4:	 @
-assignvariableop_1_sequential_mlp_mlp_0weight:	�
9
+assignvariableop_2_sequential_mlp_mlp_0bias:
+
assignvariableop_3_variable_3:
+
assignvariableop_4_variable_2:
0
assignvariableop_5_variable_1:	�
.
assignvariableop_6_variable:	�


identity_8��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB%layers/0/w/.ATTRIBUTES/VARIABLE_VALUEB%layers/0/b/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_4Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp-assignvariableop_1_sequential_mlp_mlp_0weightIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp+assignvariableop_2_sequential_mlp_mlp_0biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_3Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_2Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_1Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variableIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*
_output_shapes
 "!

identity_8Identity_8:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
Variable:*&
$
_user_specified_name
Variable_1:*&
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:84
2
_user_specified_nameSequential_mlp_mlp_0bias::6
4
_user_specified_nameSequential_mlp_mlp_0weight:*&
$
_user_specified_name
Variable_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix<
#__inference_internal_grad_fn_326170CustomGradient-325966"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
:
inputs0
serving_default_inputs:0����������<
output_00
StatefulPartitionedCall:0���������
tensorflow/serving/predict:�&
�

layers
in_dims
layer_types
	optimizer
metrics
training_messg
validation_messg
__call__
	
train_step

validation_step

signatures"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

_variables
_trainable_variables
 _trainable_variables_indices
_iterations

_momentums
_velocities"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trace_0
trace_1
trace_22�
__inference___call___325745
__inference___call___325768
__inference___call___325791�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1ztrace_2
�
trace_02�
__inference_train_step_326044�
���
FullArgSpec!
args�
	jx_batch
	jy_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_0
trace_12�
"__inference_validation_step_326076
"__inference_validation_step_326108�
���
FullArgSpec)
args!�
jx_val_batch
jy_val_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
,
serving_default"
signature_map
,
w
b"
_generic_user_object
C
0
1
2
3
4"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
:	 2adam/iteration
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�B�
__inference___call___325745inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference___call___325768inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference___call___325791inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_train_step_326044x_batchy_batch"�
���
FullArgSpec!
args�
	jx_batch
	jy_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_validation_step_326076x_val_batchy_val_batch"�
���
FullArgSpec)
args!�
jx_val_batch
jy_val_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_validation_step_326108x_val_batchy_val_batch"�
���
FullArgSpec)
args!�
jx_val_batch
jy_val_batch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_325722inputs"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jinputs
kwonlydefaults
 
annotations� *
 
-:+	�
2Sequential_mlp_mlp_0weight
&:$
2Sequential_mlp_mlp_0bias
4:2
2(adam/Sequential_mlp_mlp_0bias_0_momentum
4:2
2(adam/Sequential_mlp_mlp_0bias_0_velocity
;:9	�
2*adam/Sequential_mlp_mlp_0weight_0_momentum
;:9	�
2*adam/Sequential_mlp_mlp_0weight_0_velocityf
__inference___call___325745G'�$
�
�
inputs	 �
� "�
unknown 
f
__inference___call___325768G'�$
�
�
inputs	�
� "�
unknown
x
__inference___call___325791Y0�-
&�#
!�
inputs����������
� "!�
unknown���������
�
#__inference_internal_grad_fn_326170����
���

 
�
result_grads_0

 �
result_grads_1	�

�
result_grads_2

 �
result_grads_3	�

� ">�;

 

 
�
tensor_2

�
tensor_3	�
�
$__inference_signature_wrapper_325722u:�7
� 
0�-
+
inputs!�
inputs����������"3�0
.
output_0"�
output_0���������
�
__inference_train_step_326044�B�?
8�5
�
x_batch	 �
�
y_batch 

� "1�.
�
tensor_0 
�
tensor_1 
�
"__inference_validation_step_326076�J�G
@�=
�
x_val_batch	 �
�
y_val_batch 

� "1�.
�
tensor_0 
�
tensor_1 
�
"__inference_validation_step_326108�J�G
@�=
�
x_val_batch	�
�
y_val_batch

� "1�.
�
tensor_0 
�
tensor_1
