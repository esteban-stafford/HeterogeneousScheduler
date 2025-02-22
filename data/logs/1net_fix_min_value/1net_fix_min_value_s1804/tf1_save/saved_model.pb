��#
�'�'
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
#
	LogicalOr
x

y

z
�
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
�
Multinomial
logits"T
num_samples
output"output_dtype"
seedint "
seed2int "
Ttype:
2	" 
output_dtypetype0	:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02v1.14.0-rc1-22-gaf24dc91b5Ĥ"
r
PlaceholderPlaceholder*
dtype0*)
_output_shapes
:�����������*
shape:�����������
h
Placeholder_1Placeholder*
shape:���������*
dtype0*#
_output_shapes
:���������
r
Placeholder_2Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
h
Placeholder_3Placeholder*
dtype0*
shape:���������*#
_output_shapes
:���������
h
Placeholder_4Placeholder*#
_output_shapes
:���������*
shape:���������*
dtype0
h
Placeholder_5Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
e
pi/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"���� 
     
y

pi/ReshapeReshapePlaceholderpi/Reshape/shape*,
_output_shapes
:����������*
Tshape0*
T0
�
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"       *
dtype0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:
�
.pi/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
valueB
 *��Ⱦ*
dtype0
�
.pi/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *���>*"
_class
loc:@pi/dense/kernel
�
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
seed2*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes

: *
dtype0*
seed�
�
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@pi/dense/kernel
�
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

: 
�
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
T0
�
pi/dense/kernel
VariableV2*
_output_shapes

: *
shared_name *"
_class
loc:@pi/dense/kernel*
dtype0*
shape
: *
	container 
�
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

: *
use_locking(*"
_class
loc:@pi/dense/kernel*
T0
~
pi/dense/kernel/readIdentitypi/dense/kernel*
_output_shapes

: *
T0*"
_class
loc:@pi/dense/kernel
�
pi/dense/bias/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes
: * 
_class
loc:@pi/dense/bias
�
pi/dense/bias
VariableV2*
dtype0*
_output_shapes
: *
	container *
shared_name * 
_class
loc:@pi/dense/bias*
shape: 
�
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
use_locking(
t
pi/dense/bias/readIdentitypi/dense/bias*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
a
pi/dense/Tensordot/axesConst*
dtype0*
_output_shapes
:*
valueB:
h
pi/dense/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
b
pi/dense/Tensordot/ShapeShape
pi/Reshape*
T0*
out_type0*
_output_shapes
:
b
 pi/dense/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
pi/dense/Tensordot/GatherV2GatherV2pi/dense/Tensordot/Shapepi/dense/Tensordot/free pi/dense/Tensordot/GatherV2/axis*
Taxis0*

batch_dims *
_output_shapes
:*
Tparams0*
Tindices0
d
"pi/dense/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
pi/dense/Tensordot/GatherV2_1GatherV2pi/dense/Tensordot/Shapepi/dense/Tensordot/axes"pi/dense/Tensordot/GatherV2_1/axis*
Tparams0*
Taxis0*
_output_shapes
:*

batch_dims *
Tindices0
b
pi/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
pi/dense/Tensordot/ProdProdpi/dense/Tensordot/GatherV2pi/dense/Tensordot/Const*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
d
pi/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
pi/dense/Tensordot/Prod_1Prodpi/dense/Tensordot/GatherV2_1pi/dense/Tensordot/Const_1*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
`
pi/dense/Tensordot/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
pi/dense/Tensordot/concatConcatV2pi/dense/Tensordot/freepi/dense/Tensordot/axespi/dense/Tensordot/concat/axis*
T0*

Tidx0*
N*
_output_shapes
:
�
pi/dense/Tensordot/stackPackpi/dense/Tensordot/Prodpi/dense/Tensordot/Prod_1*

axis *
N*
_output_shapes
:*
T0
�
pi/dense/Tensordot/transpose	Transpose
pi/Reshapepi/dense/Tensordot/concat*
T0*,
_output_shapes
:����������*
Tperm0
�
pi/dense/Tensordot/ReshapeReshapepi/dense/Tensordot/transposepi/dense/Tensordot/stack*
Tshape0*0
_output_shapes
:������������������*
T0
t
#pi/dense/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
�
pi/dense/Tensordot/transpose_1	Transposepi/dense/kernel/read#pi/dense/Tensordot/transpose_1/perm*
Tperm0*
_output_shapes

: *
T0
s
"pi/dense/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
�
pi/dense/Tensordot/Reshape_1Reshapepi/dense/Tensordot/transpose_1"pi/dense/Tensordot/Reshape_1/shape*
T0*
_output_shapes

: *
Tshape0
�
pi/dense/Tensordot/MatMulMatMulpi/dense/Tensordot/Reshapepi/dense/Tensordot/Reshape_1*'
_output_shapes
:��������� *
T0*
transpose_a( *
transpose_b( 
d
pi/dense/Tensordot/Const_2Const*
_output_shapes
:*
valueB: *
dtype0
b
 pi/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
pi/dense/Tensordot/concat_1ConcatV2pi/dense/Tensordot/GatherV2pi/dense/Tensordot/Const_2 pi/dense/Tensordot/concat_1/axis*
T0*
_output_shapes
:*

Tidx0*
N
�
pi/dense/TensordotReshapepi/dense/Tensordot/MatMulpi/dense/Tensordot/concat_1*
Tshape0*
T0*,
_output_shapes
:���������� 
�
pi/dense/BiasAddBiasAddpi/dense/Tensordotpi/dense/bias/read*,
_output_shapes
:���������� *
T0*
data_formatNHWC
^
pi/dense/ReluRelupi/dense/BiasAdd*,
_output_shapes
:���������� *
T0
�
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*$
_class
loc:@pi/dense_1/kernel*
dtype0*
valueB"       
�
0pi/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *���*$
_class
loc:@pi/dense_1/kernel
�
0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *��>*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel
�
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
seed�*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel*
seed24
�
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: 
�
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: *
T0
�
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

: 
�
pi/dense_1/kernel
VariableV2*
	container *
shape
: *
shared_name *
_output_shapes

: *
dtype0*$
_class
loc:@pi/dense_1/kernel
�
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
�
pi/dense_1/kernel/readIdentitypi/dense_1/kernel*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: 
�
!pi/dense_1/bias/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
valueB*    *
dtype0
�
pi/dense_1/bias
VariableV2*
shape:*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
shared_name *
	container *
dtype0
�
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias
z
pi/dense_1/bias/readIdentitypi/dense_1/bias*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:
c
pi/dense_1/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
j
pi/dense_1/Tensordot/freeConst*
_output_shapes
:*
valueB"       *
dtype0
g
pi/dense_1/Tensordot/ShapeShapepi/dense/Relu*
_output_shapes
:*
out_type0*
T0
d
"pi/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
pi/dense_1/Tensordot/GatherV2GatherV2pi/dense_1/Tensordot/Shapepi/dense_1/Tensordot/free"pi/dense_1/Tensordot/GatherV2/axis*
Tindices0*
Taxis0*

batch_dims *
_output_shapes
:*
Tparams0
f
$pi/dense_1/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
pi/dense_1/Tensordot/GatherV2_1GatherV2pi/dense_1/Tensordot/Shapepi/dense_1/Tensordot/axes$pi/dense_1/Tensordot/GatherV2_1/axis*
Tparams0*
Tindices0*

batch_dims *
_output_shapes
:*
Taxis0
d
pi/dense_1/Tensordot/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
pi/dense_1/Tensordot/ProdProdpi/dense_1/Tensordot/GatherV2pi/dense_1/Tensordot/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
f
pi/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
pi/dense_1/Tensordot/Prod_1Prodpi/dense_1/Tensordot/GatherV2_1pi/dense_1/Tensordot/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
b
 pi/dense_1/Tensordot/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
pi/dense_1/Tensordot/concatConcatV2pi/dense_1/Tensordot/freepi/dense_1/Tensordot/axes pi/dense_1/Tensordot/concat/axis*
T0*

Tidx0*
N*
_output_shapes
:
�
pi/dense_1/Tensordot/stackPackpi/dense_1/Tensordot/Prodpi/dense_1/Tensordot/Prod_1*

axis *
N*
T0*
_output_shapes
:
�
pi/dense_1/Tensordot/transpose	Transposepi/dense/Relupi/dense_1/Tensordot/concat*
T0*
Tperm0*,
_output_shapes
:���������� 
�
pi/dense_1/Tensordot/ReshapeReshapepi/dense_1/Tensordot/transposepi/dense_1/Tensordot/stack*
T0*0
_output_shapes
:������������������*
Tshape0
v
%pi/dense_1/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
 pi/dense_1/Tensordot/transpose_1	Transposepi/dense_1/kernel/read%pi/dense_1/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

: 
u
$pi/dense_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
valueB"       *
dtype0
�
pi/dense_1/Tensordot/Reshape_1Reshape pi/dense_1/Tensordot/transpose_1$pi/dense_1/Tensordot/Reshape_1/shape*
Tshape0*
_output_shapes

: *
T0
�
pi/dense_1/Tensordot/MatMulMatMulpi/dense_1/Tensordot/Reshapepi/dense_1/Tensordot/Reshape_1*'
_output_shapes
:���������*
T0*
transpose_b( *
transpose_a( 
f
pi/dense_1/Tensordot/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
d
"pi/dense_1/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
pi/dense_1/Tensordot/concat_1ConcatV2pi/dense_1/Tensordot/GatherV2pi/dense_1/Tensordot/Const_2"pi/dense_1/Tensordot/concat_1/axis*
_output_shapes
:*
T0*

Tidx0*
N
�
pi/dense_1/TensordotReshapepi/dense_1/Tensordot/MatMulpi/dense_1/Tensordot/concat_1*,
_output_shapes
:����������*
Tshape0*
T0
�
pi/dense_1/BiasAddBiasAddpi/dense_1/Tensordotpi/dense_1/bias/read*
data_formatNHWC*,
_output_shapes
:����������*
T0
b
pi/dense_1/ReluRelupi/dense_1/BiasAdd*
T0*,
_output_shapes
:����������
�
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*$
_class
loc:@pi/dense_2/kernel*
valueB"      
�
0pi/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *   �*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
dtype0
�
0pi/dense_2/kernel/Initializer/random_uniform/maxConst*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: *
dtype0*
valueB
 *   ?
�
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*$
_class
loc:@pi/dense_2/kernel*
seed2]*
dtype0*
seed�*
_output_shapes

:*
T0
�
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: 
�
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:*
T0
�
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:
�
pi/dense_2/kernel
VariableV2*
dtype0*$
_class
loc:@pi/dense_2/kernel*
	container *
shape
:*
shared_name *
_output_shapes

:
�
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:*
use_locking(
�
pi/dense_2/kernel/readIdentitypi/dense_2/kernel*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:*
T0
�
!pi/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0
�
pi/dense_2/bias
VariableV2*
	container *"
_class
loc:@pi/dense_2/bias*
shared_name *
dtype0*
shape:*
_output_shapes
:
�
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
c
pi/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
j
pi/dense_2/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       
i
pi/dense_2/Tensordot/ShapeShapepi/dense_1/Relu*
out_type0*
_output_shapes
:*
T0
d
"pi/dense_2/Tensordot/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
pi/dense_2/Tensordot/GatherV2GatherV2pi/dense_2/Tensordot/Shapepi/dense_2/Tensordot/free"pi/dense_2/Tensordot/GatherV2/axis*
Tparams0*
Tindices0*

batch_dims *
_output_shapes
:*
Taxis0
f
$pi/dense_2/Tensordot/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
pi/dense_2/Tensordot/GatherV2_1GatherV2pi/dense_2/Tensordot/Shapepi/dense_2/Tensordot/axes$pi/dense_2/Tensordot/GatherV2_1/axis*
Tparams0*

batch_dims *
_output_shapes
:*
Taxis0*
Tindices0
d
pi/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
pi/dense_2/Tensordot/ProdProdpi/dense_2/Tensordot/GatherV2pi/dense_2/Tensordot/Const*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
f
pi/dense_2/Tensordot/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
pi/dense_2/Tensordot/Prod_1Prodpi/dense_2/Tensordot/GatherV2_1pi/dense_2/Tensordot/Const_1*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
b
 pi/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
pi/dense_2/Tensordot/concatConcatV2pi/dense_2/Tensordot/freepi/dense_2/Tensordot/axes pi/dense_2/Tensordot/concat/axis*

Tidx0*
_output_shapes
:*
T0*
N
�
pi/dense_2/Tensordot/stackPackpi/dense_2/Tensordot/Prodpi/dense_2/Tensordot/Prod_1*
_output_shapes
:*
T0*
N*

axis 
�
pi/dense_2/Tensordot/transpose	Transposepi/dense_1/Relupi/dense_2/Tensordot/concat*,
_output_shapes
:����������*
Tperm0*
T0
�
pi/dense_2/Tensordot/ReshapeReshapepi/dense_2/Tensordot/transposepi/dense_2/Tensordot/stack*0
_output_shapes
:������������������*
Tshape0*
T0
v
%pi/dense_2/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       
�
 pi/dense_2/Tensordot/transpose_1	Transposepi/dense_2/kernel/read%pi/dense_2/Tensordot/transpose_1/perm*
Tperm0*
T0*
_output_shapes

:
u
$pi/dense_2/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
�
pi/dense_2/Tensordot/Reshape_1Reshape pi/dense_2/Tensordot/transpose_1$pi/dense_2/Tensordot/Reshape_1/shape*
_output_shapes

:*
Tshape0*
T0
�
pi/dense_2/Tensordot/MatMulMatMulpi/dense_2/Tensordot/Reshapepi/dense_2/Tensordot/Reshape_1*
transpose_a( *'
_output_shapes
:���������*
transpose_b( *
T0
f
pi/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
d
"pi/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
pi/dense_2/Tensordot/concat_1ConcatV2pi/dense_2/Tensordot/GatherV2pi/dense_2/Tensordot/Const_2"pi/dense_2/Tensordot/concat_1/axis*

Tidx0*
T0*
_output_shapes
:*
N
�
pi/dense_2/TensordotReshapepi/dense_2/Tensordot/MatMulpi/dense_2/Tensordot/concat_1*,
_output_shapes
:����������*
Tshape0*
T0
�
pi/dense_2/BiasAddBiasAddpi/dense_2/Tensordotpi/dense_2/bias/read*,
_output_shapes
:����������*
data_formatNHWC*
T0
b
pi/dense_2/ReluRelupi/dense_2/BiasAdd*
T0*,
_output_shapes
:����������
�
2pi/dense_3/kernel/Initializer/random_uniform/shapeConst*
valueB"      *$
_class
loc:@pi/dense_3/kernel*
_output_shapes
:*
dtype0
�
0pi/dense_3/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *$
_class
loc:@pi/dense_3/kernel*
valueB
 *�Q�*
dtype0
�
0pi/dense_3/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@pi/dense_3/kernel*
valueB
 *�Q?*
dtype0
�
:pi/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_3/kernel/Initializer/random_uniform/shape*
dtype0*
seed2�*
seed�*
T0*
_output_shapes

:*$
_class
loc:@pi/dense_3/kernel
�
0pi/dense_3/kernel/Initializer/random_uniform/subSub0pi/dense_3/kernel/Initializer/random_uniform/max0pi/dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *$
_class
loc:@pi/dense_3/kernel*
T0
�
0pi/dense_3/kernel/Initializer/random_uniform/mulMul:pi/dense_3/kernel/Initializer/random_uniform/RandomUniform0pi/dense_3/kernel/Initializer/random_uniform/sub*
_output_shapes

:*$
_class
loc:@pi/dense_3/kernel*
T0
�
,pi/dense_3/kernel/Initializer/random_uniformAdd0pi/dense_3/kernel/Initializer/random_uniform/mul0pi/dense_3/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:*$
_class
loc:@pi/dense_3/kernel
�
pi/dense_3/kernel
VariableV2*
	container *$
_class
loc:@pi/dense_3/kernel*
shared_name *
_output_shapes

:*
shape
:*
dtype0
�
pi/dense_3/kernel/AssignAssignpi/dense_3/kernel,pi/dense_3/kernel/Initializer/random_uniform*
_output_shapes

:*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_3/kernel
�
pi/dense_3/kernel/readIdentitypi/dense_3/kernel*
_output_shapes

:*
T0*$
_class
loc:@pi/dense_3/kernel
�
!pi/dense_3/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:*
valueB*    
�
pi/dense_3/bias
VariableV2*
shared_name *"
_class
loc:@pi/dense_3/bias*
shape:*
_output_shapes
:*
dtype0*
	container 
�
pi/dense_3/bias/AssignAssignpi/dense_3/bias!pi/dense_3/bias/Initializer/zeros*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_3/bias
z
pi/dense_3/bias/readIdentitypi/dense_3/bias*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias
c
pi/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
j
pi/dense_3/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       
i
pi/dense_3/Tensordot/ShapeShapepi/dense_2/Relu*
out_type0*
_output_shapes
:*
T0
d
"pi/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
pi/dense_3/Tensordot/GatherV2GatherV2pi/dense_3/Tensordot/Shapepi/dense_3/Tensordot/free"pi/dense_3/Tensordot/GatherV2/axis*
Tindices0*

batch_dims *
Taxis0*
Tparams0*
_output_shapes
:
f
$pi/dense_3/Tensordot/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
pi/dense_3/Tensordot/GatherV2_1GatherV2pi/dense_3/Tensordot/Shapepi/dense_3/Tensordot/axes$pi/dense_3/Tensordot/GatherV2_1/axis*
Tparams0*

batch_dims *
Taxis0*
_output_shapes
:*
Tindices0
d
pi/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
pi/dense_3/Tensordot/ProdProdpi/dense_3/Tensordot/GatherV2pi/dense_3/Tensordot/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
f
pi/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
pi/dense_3/Tensordot/Prod_1Prodpi/dense_3/Tensordot/GatherV2_1pi/dense_3/Tensordot/Const_1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
b
 pi/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
pi/dense_3/Tensordot/concatConcatV2pi/dense_3/Tensordot/freepi/dense_3/Tensordot/axes pi/dense_3/Tensordot/concat/axis*
_output_shapes
:*
N*
T0*

Tidx0
�
pi/dense_3/Tensordot/stackPackpi/dense_3/Tensordot/Prodpi/dense_3/Tensordot/Prod_1*
_output_shapes
:*
N*
T0*

axis 
�
pi/dense_3/Tensordot/transpose	Transposepi/dense_2/Relupi/dense_3/Tensordot/concat*
Tperm0*,
_output_shapes
:����������*
T0
�
pi/dense_3/Tensordot/ReshapeReshapepi/dense_3/Tensordot/transposepi/dense_3/Tensordot/stack*
Tshape0*
T0*0
_output_shapes
:������������������
v
%pi/dense_3/Tensordot/transpose_1/permConst*
dtype0*
valueB"       *
_output_shapes
:
�
 pi/dense_3/Tensordot/transpose_1	Transposepi/dense_3/kernel/read%pi/dense_3/Tensordot/transpose_1/perm*
Tperm0*
_output_shapes

:*
T0
u
$pi/dense_3/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
pi/dense_3/Tensordot/Reshape_1Reshape pi/dense_3/Tensordot/transpose_1$pi/dense_3/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:
�
pi/dense_3/Tensordot/MatMulMatMulpi/dense_3/Tensordot/Reshapepi/dense_3/Tensordot/Reshape_1*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
f
pi/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
valueB:*
dtype0
d
"pi/dense_3/Tensordot/concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
pi/dense_3/Tensordot/concat_1ConcatV2pi/dense_3/Tensordot/GatherV2pi/dense_3/Tensordot/Const_2"pi/dense_3/Tensordot/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
pi/dense_3/TensordotReshapepi/dense_3/Tensordot/MatMulpi/dense_3/Tensordot/concat_1*
Tshape0*,
_output_shapes
:����������*
T0
�
pi/dense_3/BiasAddBiasAddpi/dense_3/Tensordotpi/dense_3/bias/read*
data_formatNHWC*,
_output_shapes
:����������*
T0
|

pi/SqueezeSqueezepi/dense_3/BiasAdd*
squeeze_dims

���������*
T0*(
_output_shapes
:����������
M
pi/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
Y
pi/subSubPlaceholder_2pi/sub/y*
T0*(
_output_shapes
:����������
M
pi/mul/yConst*
dtype0*
valueB
 * $tI*
_output_shapes
: 
R
pi/mulMulpi/subpi/mul/y*(
_output_shapes
:����������*
T0
T
pi/addAdd
pi/Squeezepi/mul*
T0*(
_output_shapes
:����������
V
pi/LogSoftmax
LogSoftmaxpi/add*
T0*(
_output_shapes
:����������
h
&pi/multinomial/Multinomial/num_samplesConst*
_output_shapes
: *
value	B :*
dtype0
�
pi/multinomial/MultinomialMultinomialpi/add&pi/multinomial/Multinomial/num_samples*
T0*
seed�*
output_dtype0	*
seed2�*'
_output_shapes
:���������
x
pi/Squeeze_1Squeezepi/multinomial/Multinomial*
squeeze_dims
*#
_output_shapes
:���������*
T0	
X
pi/one_hot/on_valueConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
Y
pi/one_hot/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
S
pi/one_hot/depthConst*
value
B :�*
_output_shapes
: *
dtype0
�

pi/one_hotOneHotPlaceholder_1pi/one_hot/depthpi/one_hot/on_valuepi/one_hot/off_value*
T0*
TI0*(
_output_shapes
:����������*
axis���������
]
pi/mul_1Mul
pi/one_hotpi/LogSoftmax*
T0*(
_output_shapes
:����������
Z
pi/Sum/reduction_indicesConst*
value	B :*
_output_shapes
: *
dtype0
|
pi/SumSumpi/mul_1pi/Sum/reduction_indices*
	keep_dims( *
T0*#
_output_shapes
:���������*

Tidx0
Z
pi/one_hot_1/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
[
pi/one_hot_1/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
U
pi/one_hot_1/depthConst*
value
B :�*
dtype0*
_output_shapes
: 
�
pi/one_hot_1OneHotpi/Squeeze_1pi/one_hot_1/depthpi/one_hot_1/on_valuepi/one_hot_1/off_value*(
_output_shapes
:����������*
axis���������*
TI0	*
T0
_
pi/mul_2Mulpi/one_hot_1pi/LogSoftmax*
T0*(
_output_shapes
:����������
\
pi/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
pi/Sum_1Sumpi/mul_2pi/Sum_1/reduction_indices*
	keep_dims( *#
_output_shapes
:���������*

Tidx0*
T0
d
v/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"���� 
     
w
	v/ReshapeReshapePlaceholderv/Reshape/shape*
Tshape0*,
_output_shapes
:����������*
T0
�
/v/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*!
_class
loc:@v/dense/kernel*
valueB"       
�
-v/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *��Ⱦ*!
_class
loc:@v/dense/kernel*
_output_shapes
: *
dtype0
�
-v/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *���>*!
_class
loc:@v/dense/kernel
�
7v/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform/v/dense/kernel/Initializer/random_uniform/shape*
seed�*
dtype0*
_output_shapes

: *!
_class
loc:@v/dense/kernel*
seed2�*
T0
�
-v/dense/kernel/Initializer/random_uniform/subSub-v/dense/kernel/Initializer/random_uniform/max-v/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@v/dense/kernel
�
-v/dense/kernel/Initializer/random_uniform/mulMul7v/dense/kernel/Initializer/random_uniform/RandomUniform-v/dense/kernel/Initializer/random_uniform/sub*
_output_shapes

: *
T0*!
_class
loc:@v/dense/kernel
�
)v/dense/kernel/Initializer/random_uniformAdd-v/dense/kernel/Initializer/random_uniform/mul-v/dense/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

: 
�
v/dense/kernel
VariableV2*!
_class
loc:@v/dense/kernel*
dtype0*
shared_name *
shape
: *
	container *
_output_shapes

: 
�
v/dense/kernel/AssignAssignv/dense/kernel)v/dense/kernel/Initializer/random_uniform*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

: *
T0*
use_locking(
{
v/dense/kernel/readIdentityv/dense/kernel*
T0*
_output_shapes

: *!
_class
loc:@v/dense/kernel
�
v/dense/bias/Initializer/zerosConst*
_output_shapes
: *
dtype0*
_class
loc:@v/dense/bias*
valueB *    
�
v/dense/bias
VariableV2*
_class
loc:@v/dense/bias*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
v/dense/bias/AssignAssignv/dense/biasv/dense/bias/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
q
v/dense/bias/readIdentityv/dense/bias*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
`
v/dense/Tensordot/axesConst*
_output_shapes
:*
valueB:*
dtype0
g
v/dense/Tensordot/freeConst*
dtype0*
valueB"       *
_output_shapes
:
`
v/dense/Tensordot/ShapeShape	v/Reshape*
_output_shapes
:*
out_type0*
T0
a
v/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
v/dense/Tensordot/GatherV2GatherV2v/dense/Tensordot/Shapev/dense/Tensordot/freev/dense/Tensordot/GatherV2/axis*
Tparams0*
Tindices0*
Taxis0*
_output_shapes
:*

batch_dims 
c
!v/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
v/dense/Tensordot/GatherV2_1GatherV2v/dense/Tensordot/Shapev/dense/Tensordot/axes!v/dense/Tensordot/GatherV2_1/axis*

batch_dims *
Tindices0*
Taxis0*
_output_shapes
:*
Tparams0
a
v/dense/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
v/dense/Tensordot/ProdProdv/dense/Tensordot/GatherV2v/dense/Tensordot/Const*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
c
v/dense/Tensordot/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
v/dense/Tensordot/Prod_1Prodv/dense/Tensordot/GatherV2_1v/dense/Tensordot/Const_1*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
_
v/dense/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
v/dense/Tensordot/concatConcatV2v/dense/Tensordot/freev/dense/Tensordot/axesv/dense/Tensordot/concat/axis*
_output_shapes
:*
T0*
N*

Tidx0
�
v/dense/Tensordot/stackPackv/dense/Tensordot/Prodv/dense/Tensordot/Prod_1*
T0*
_output_shapes
:*
N*

axis 
�
v/dense/Tensordot/transpose	Transpose	v/Reshapev/dense/Tensordot/concat*,
_output_shapes
:����������*
T0*
Tperm0
�
v/dense/Tensordot/ReshapeReshapev/dense/Tensordot/transposev/dense/Tensordot/stack*0
_output_shapes
:������������������*
Tshape0*
T0
s
"v/dense/Tensordot/transpose_1/permConst*
valueB"       *
_output_shapes
:*
dtype0
�
v/dense/Tensordot/transpose_1	Transposev/dense/kernel/read"v/dense/Tensordot/transpose_1/perm*
T0*
_output_shapes

: *
Tperm0
r
!v/dense/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
valueB"       *
dtype0
�
v/dense/Tensordot/Reshape_1Reshapev/dense/Tensordot/transpose_1!v/dense/Tensordot/Reshape_1/shape*
_output_shapes

: *
Tshape0*
T0
�
v/dense/Tensordot/MatMulMatMulv/dense/Tensordot/Reshapev/dense/Tensordot/Reshape_1*
transpose_a( *
T0*'
_output_shapes
:��������� *
transpose_b( 
c
v/dense/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB: 
a
v/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
v/dense/Tensordot/concat_1ConcatV2v/dense/Tensordot/GatherV2v/dense/Tensordot/Const_2v/dense/Tensordot/concat_1/axis*

Tidx0*
T0*
_output_shapes
:*
N
�
v/dense/TensordotReshapev/dense/Tensordot/MatMulv/dense/Tensordot/concat_1*
Tshape0*
T0*,
_output_shapes
:���������� 
�
v/dense/BiasAddBiasAddv/dense/Tensordotv/dense/bias/read*
T0*,
_output_shapes
:���������� *
data_formatNHWC
\
v/dense/ReluReluv/dense/BiasAdd*
T0*,
_output_shapes
:���������� 
�
1v/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*#
_class
loc:@v/dense_1/kernel*
valueB"       
�
/v/dense_1/kernel/Initializer/random_uniform/minConst*#
_class
loc:@v/dense_1/kernel*
valueB
 *���*
_output_shapes
: *
dtype0
�
/v/dense_1/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@v/dense_1/kernel*
valueB
 *��>*
dtype0*
_output_shapes
: 
�
9v/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_1/kernel/Initializer/random_uniform/shape*
_output_shapes

: *
seed�*
seed2�*
T0*#
_class
loc:@v/dense_1/kernel*
dtype0
�
/v/dense_1/kernel/Initializer/random_uniform/subSub/v/dense_1/kernel/Initializer/random_uniform/max/v/dense_1/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_1/kernel*
_output_shapes
: *
T0
�
/v/dense_1/kernel/Initializer/random_uniform/mulMul9v/dense_1/kernel/Initializer/random_uniform/RandomUniform/v/dense_1/kernel/Initializer/random_uniform/sub*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
T0
�
+v/dense_1/kernel/Initializer/random_uniformAdd/v/dense_1/kernel/Initializer/random_uniform/mul/v/dense_1/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
T0
�
v/dense_1/kernel
VariableV2*
shape
: *
shared_name *
_output_shapes

: *
dtype0*#
_class
loc:@v/dense_1/kernel*
	container 
�
v/dense_1/kernel/AssignAssignv/dense_1/kernel+v/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(
�
v/dense_1/kernel/readIdentityv/dense_1/kernel*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

: 
�
 v/dense_1/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
dtype0*!
_class
loc:@v/dense_1/bias
�
v/dense_1/bias
VariableV2*
dtype0*
shared_name *!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
shape:*
	container 
�
v/dense_1/bias/AssignAssignv/dense_1/bias v/dense_1/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0
w
v/dense_1/bias/readIdentityv/dense_1/bias*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:
b
v/dense_1/Tensordot/axesConst*
_output_shapes
:*
valueB:*
dtype0
i
v/dense_1/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       
e
v/dense_1/Tensordot/ShapeShapev/dense/Relu*
_output_shapes
:*
out_type0*
T0
c
!v/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
v/dense_1/Tensordot/GatherV2GatherV2v/dense_1/Tensordot/Shapev/dense_1/Tensordot/free!v/dense_1/Tensordot/GatherV2/axis*

batch_dims *
Taxis0*
Tparams0*
Tindices0*
_output_shapes
:
e
#v/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
v/dense_1/Tensordot/GatherV2_1GatherV2v/dense_1/Tensordot/Shapev/dense_1/Tensordot/axes#v/dense_1/Tensordot/GatherV2_1/axis*
Taxis0*
_output_shapes
:*
Tparams0*
Tindices0*

batch_dims 
c
v/dense_1/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
v/dense_1/Tensordot/ProdProdv/dense_1/Tensordot/GatherV2v/dense_1/Tensordot/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
e
v/dense_1/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
v/dense_1/Tensordot/Prod_1Prodv/dense_1/Tensordot/GatherV2_1v/dense_1/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
a
v/dense_1/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
v/dense_1/Tensordot/concatConcatV2v/dense_1/Tensordot/freev/dense_1/Tensordot/axesv/dense_1/Tensordot/concat/axis*
T0*
_output_shapes
:*
N*

Tidx0
�
v/dense_1/Tensordot/stackPackv/dense_1/Tensordot/Prodv/dense_1/Tensordot/Prod_1*
_output_shapes
:*
T0*

axis *
N
�
v/dense_1/Tensordot/transpose	Transposev/dense/Reluv/dense_1/Tensordot/concat*
T0*
Tperm0*,
_output_shapes
:���������� 
�
v/dense_1/Tensordot/ReshapeReshapev/dense_1/Tensordot/transposev/dense_1/Tensordot/stack*
Tshape0*
T0*0
_output_shapes
:������������������
u
$v/dense_1/Tensordot/transpose_1/permConst*
dtype0*
valueB"       *
_output_shapes
:
�
v/dense_1/Tensordot/transpose_1	Transposev/dense_1/kernel/read$v/dense_1/Tensordot/transpose_1/perm*
T0*
_output_shapes

: *
Tperm0
t
#v/dense_1/Tensordot/Reshape_1/shapeConst*
valueB"       *
_output_shapes
:*
dtype0
�
v/dense_1/Tensordot/Reshape_1Reshapev/dense_1/Tensordot/transpose_1#v/dense_1/Tensordot/Reshape_1/shape*
_output_shapes

: *
Tshape0*
T0
�
v/dense_1/Tensordot/MatMulMatMulv/dense_1/Tensordot/Reshapev/dense_1/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b( *'
_output_shapes
:���������
e
v/dense_1/Tensordot/Const_2Const*
valueB:*
_output_shapes
:*
dtype0
c
!v/dense_1/Tensordot/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
v/dense_1/Tensordot/concat_1ConcatV2v/dense_1/Tensordot/GatherV2v/dense_1/Tensordot/Const_2!v/dense_1/Tensordot/concat_1/axis*
N*
_output_shapes
:*
T0*

Tidx0
�
v/dense_1/TensordotReshapev/dense_1/Tensordot/MatMulv/dense_1/Tensordot/concat_1*
Tshape0*
T0*,
_output_shapes
:����������
�
v/dense_1/BiasAddBiasAddv/dense_1/Tensordotv/dense_1/bias/read*,
_output_shapes
:����������*
data_formatNHWC*
T0
`
v/dense_1/ReluReluv/dense_1/BiasAdd*,
_output_shapes
:����������*
T0
�
1v/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*#
_class
loc:@v/dense_2/kernel*
dtype0*
valueB"      
�
/v/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *   �*#
_class
loc:@v/dense_2/kernel*
dtype0*
_output_shapes
: 
�
/v/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *   ?*#
_class
loc:@v/dense_2/kernel*
dtype0*
_output_shapes
: 
�
9v/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_2/kernel/Initializer/random_uniform/shape*
seed�*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:*
seed2�*
dtype0
�
/v/dense_2/kernel/Initializer/random_uniform/subSub/v/dense_2/kernel/Initializer/random_uniform/max/v/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *#
_class
loc:@v/dense_2/kernel*
T0
�
/v/dense_2/kernel/Initializer/random_uniform/mulMul9v/dense_2/kernel/Initializer/random_uniform/RandomUniform/v/dense_2/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:
�
+v/dense_2/kernel/Initializer/random_uniformAdd/v/dense_2/kernel/Initializer/random_uniform/mul/v/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
T0
�
v/dense_2/kernel
VariableV2*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
shared_name *
shape
:*
dtype0*
	container 
�
v/dense_2/kernel/AssignAssignv/dense_2/kernel+v/dense_2/kernel/Initializer/random_uniform*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
T0*
use_locking(
�
v/dense_2/kernel/readIdentityv/dense_2/kernel*
T0*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel
�
 v/dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *!
_class
loc:@v/dense_2/bias
�
v/dense_2/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container *!
_class
loc:@v/dense_2/bias
�
v/dense_2/bias/AssignAssignv/dense_2/bias v/dense_2/bias/Initializer/zeros*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0*
validate_shape(
w
v/dense_2/bias/readIdentityv/dense_2/bias*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
b
v/dense_2/Tensordot/axesConst*
dtype0*
valueB:*
_output_shapes
:
i
v/dense_2/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       
g
v/dense_2/Tensordot/ShapeShapev/dense_1/Relu*
out_type0*
T0*
_output_shapes
:
c
!v/dense_2/Tensordot/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
v/dense_2/Tensordot/GatherV2GatherV2v/dense_2/Tensordot/Shapev/dense_2/Tensordot/free!v/dense_2/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*

batch_dims *
Tparams0*
_output_shapes
:
e
#v/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
v/dense_2/Tensordot/GatherV2_1GatherV2v/dense_2/Tensordot/Shapev/dense_2/Tensordot/axes#v/dense_2/Tensordot/GatherV2_1/axis*
Tindices0*

batch_dims *
Taxis0*
_output_shapes
:*
Tparams0
c
v/dense_2/Tensordot/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
v/dense_2/Tensordot/ProdProdv/dense_2/Tensordot/GatherV2v/dense_2/Tensordot/Const*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
e
v/dense_2/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
v/dense_2/Tensordot/Prod_1Prodv/dense_2/Tensordot/GatherV2_1v/dense_2/Tensordot/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
a
v/dense_2/Tensordot/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
v/dense_2/Tensordot/concatConcatV2v/dense_2/Tensordot/freev/dense_2/Tensordot/axesv/dense_2/Tensordot/concat/axis*
_output_shapes
:*
N*

Tidx0*
T0
�
v/dense_2/Tensordot/stackPackv/dense_2/Tensordot/Prodv/dense_2/Tensordot/Prod_1*

axis *
_output_shapes
:*
N*
T0
�
v/dense_2/Tensordot/transpose	Transposev/dense_1/Reluv/dense_2/Tensordot/concat*,
_output_shapes
:����������*
T0*
Tperm0
�
v/dense_2/Tensordot/ReshapeReshapev/dense_2/Tensordot/transposev/dense_2/Tensordot/stack*
Tshape0*
T0*0
_output_shapes
:������������������
u
$v/dense_2/Tensordot/transpose_1/permConst*
valueB"       *
_output_shapes
:*
dtype0
�
v/dense_2/Tensordot/transpose_1	Transposev/dense_2/kernel/read$v/dense_2/Tensordot/transpose_1/perm*
_output_shapes

:*
Tperm0*
T0
t
#v/dense_2/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
v/dense_2/Tensordot/Reshape_1Reshapev/dense_2/Tensordot/transpose_1#v/dense_2/Tensordot/Reshape_1/shape*
_output_shapes

:*
Tshape0*
T0
�
v/dense_2/Tensordot/MatMulMatMulv/dense_2/Tensordot/Reshapev/dense_2/Tensordot/Reshape_1*'
_output_shapes
:���������*
T0*
transpose_a( *
transpose_b( 
e
v/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
c
!v/dense_2/Tensordot/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
v/dense_2/Tensordot/concat_1ConcatV2v/dense_2/Tensordot/GatherV2v/dense_2/Tensordot/Const_2!v/dense_2/Tensordot/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
v/dense_2/TensordotReshapev/dense_2/Tensordot/MatMulv/dense_2/Tensordot/concat_1*,
_output_shapes
:����������*
T0*
Tshape0
�
v/dense_2/BiasAddBiasAddv/dense_2/Tensordotv/dense_2/bias/read*,
_output_shapes
:����������*
T0*
data_formatNHWC
`
v/dense_2/ReluReluv/dense_2/BiasAdd*
T0*,
_output_shapes
:����������
�
1v/dense_3/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@v/dense_3/kernel*
_output_shapes
:*
valueB"      *
dtype0
�
/v/dense_3/kernel/Initializer/random_uniform/minConst*#
_class
loc:@v/dense_3/kernel*
valueB
 *�Q�*
_output_shapes
: *
dtype0
�
/v/dense_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *�Q?*#
_class
loc:@v/dense_3/kernel*
dtype0*
_output_shapes
: 
�
9v/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_3/kernel/Initializer/random_uniform/shape*
T0*#
_class
loc:@v/dense_3/kernel*
dtype0*
seed2�*
_output_shapes

:*
seed�
�
/v/dense_3/kernel/Initializer/random_uniform/subSub/v/dense_3/kernel/Initializer/random_uniform/max/v/dense_3/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@v/dense_3/kernel
�
/v/dense_3/kernel/Initializer/random_uniform/mulMul9v/dense_3/kernel/Initializer/random_uniform/RandomUniform/v/dense_3/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:
�
+v/dense_3/kernel/Initializer/random_uniformAdd/v/dense_3/kernel/Initializer/random_uniform/mul/v/dense_3/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel
�
v/dense_3/kernel
VariableV2*
_output_shapes

:*
	container *
shape
:*#
_class
loc:@v/dense_3/kernel*
dtype0*
shared_name 
�
v/dense_3/kernel/AssignAssignv/dense_3/kernel+v/dense_3/kernel/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel
�
v/dense_3/kernel/readIdentityv/dense_3/kernel*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
T0
�
 v/dense_3/bias/Initializer/zerosConst*!
_class
loc:@v/dense_3/bias*
valueB*    *
dtype0*
_output_shapes
:
�
v/dense_3/bias
VariableV2*
shape:*!
_class
loc:@v/dense_3/bias*
	container *
_output_shapes
:*
shared_name *
dtype0
�
v/dense_3/bias/AssignAssignv/dense_3/bias v/dense_3/bias/Initializer/zeros*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_3/bias*
validate_shape(
w
v/dense_3/bias/readIdentityv/dense_3/bias*!
_class
loc:@v/dense_3/bias*
T0*
_output_shapes
:
b
v/dense_3/Tensordot/axesConst*
valueB:*
_output_shapes
:*
dtype0
i
v/dense_3/Tensordot/freeConst*
dtype0*
valueB"       *
_output_shapes
:
g
v/dense_3/Tensordot/ShapeShapev/dense_2/Relu*
_output_shapes
:*
out_type0*
T0
c
!v/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
v/dense_3/Tensordot/GatherV2GatherV2v/dense_3/Tensordot/Shapev/dense_3/Tensordot/free!v/dense_3/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*

batch_dims *
_output_shapes
:*
Tparams0
e
#v/dense_3/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
v/dense_3/Tensordot/GatherV2_1GatherV2v/dense_3/Tensordot/Shapev/dense_3/Tensordot/axes#v/dense_3/Tensordot/GatherV2_1/axis*
Tindices0*

batch_dims *
_output_shapes
:*
Taxis0*
Tparams0
c
v/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
v/dense_3/Tensordot/ProdProdv/dense_3/Tensordot/GatherV2v/dense_3/Tensordot/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
v/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
v/dense_3/Tensordot/Prod_1Prodv/dense_3/Tensordot/GatherV2_1v/dense_3/Tensordot/Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
a
v/dense_3/Tensordot/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
v/dense_3/Tensordot/concatConcatV2v/dense_3/Tensordot/freev/dense_3/Tensordot/axesv/dense_3/Tensordot/concat/axis*
_output_shapes
:*
N*

Tidx0*
T0
�
v/dense_3/Tensordot/stackPackv/dense_3/Tensordot/Prodv/dense_3/Tensordot/Prod_1*
_output_shapes
:*

axis *
N*
T0
�
v/dense_3/Tensordot/transpose	Transposev/dense_2/Reluv/dense_3/Tensordot/concat*
T0*
Tperm0*,
_output_shapes
:����������
�
v/dense_3/Tensordot/ReshapeReshapev/dense_3/Tensordot/transposev/dense_3/Tensordot/stack*
T0*0
_output_shapes
:������������������*
Tshape0
u
$v/dense_3/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       
�
v/dense_3/Tensordot/transpose_1	Transposev/dense_3/kernel/read$v/dense_3/Tensordot/transpose_1/perm*
_output_shapes

:*
Tperm0*
T0
t
#v/dense_3/Tensordot/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
v/dense_3/Tensordot/Reshape_1Reshapev/dense_3/Tensordot/transpose_1#v/dense_3/Tensordot/Reshape_1/shape*
Tshape0*
_output_shapes

:*
T0
�
v/dense_3/Tensordot/MatMulMatMulv/dense_3/Tensordot/Reshapev/dense_3/Tensordot/Reshape_1*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:���������
e
v/dense_3/Tensordot/Const_2Const*
valueB:*
_output_shapes
:*
dtype0
c
!v/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
v/dense_3/Tensordot/concat_1ConcatV2v/dense_3/Tensordot/GatherV2v/dense_3/Tensordot/Const_2!v/dense_3/Tensordot/concat_1/axis*
N*
T0*
_output_shapes
:*

Tidx0
�
v/dense_3/TensordotReshapev/dense_3/Tensordot/MatMulv/dense_3/Tensordot/concat_1*,
_output_shapes
:����������*
Tshape0*
T0
�
v/dense_3/BiasAddBiasAddv/dense_3/Tensordotv/dense_3/bias/read*
T0*
data_formatNHWC*,
_output_shapes
:����������
z
	v/SqueezeSqueezev/dense_3/BiasAdd*(
_output_shapes
:����������*
squeeze_dims

���������*
T0
�
1v/dense_4/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB" 
  @   *#
_class
loc:@v/dense_4/kernel
�
/v/dense_4/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *�C�*#
_class
loc:@v/dense_4/kernel*
_output_shapes
: 
�
/v/dense_4/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *�C=*#
_class
loc:@v/dense_4/kernel
�
9v/dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_4/kernel/Initializer/random_uniform/shape*
T0*
seed2�*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*
dtype0*
seed�
�
/v/dense_4/kernel/Initializer/random_uniform/subSub/v/dense_4/kernel/Initializer/random_uniform/max/v/dense_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@v/dense_4/kernel
�
/v/dense_4/kernel/Initializer/random_uniform/mulMul9v/dense_4/kernel/Initializer/random_uniform/RandomUniform/v/dense_4/kernel/Initializer/random_uniform/sub*#
_class
loc:@v/dense_4/kernel*
T0*
_output_shapes
:	�@
�
+v/dense_4/kernel/Initializer/random_uniformAdd/v/dense_4/kernel/Initializer/random_uniform/mul/v/dense_4/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@
�
v/dense_4/kernel
VariableV2*
	container *
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
dtype0*
shared_name *
shape:	�@
�
v/dense_4/kernel/AssignAssignv/dense_4/kernel+v/dense_4/kernel/Initializer/random_uniform*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
T0*
validate_shape(*
use_locking(
�
v/dense_4/kernel/readIdentityv/dense_4/kernel*
T0*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@
�
 v/dense_4/bias/Initializer/zerosConst*
dtype0*
valueB@*    *
_output_shapes
:@*!
_class
loc:@v/dense_4/bias
�
v/dense_4/bias
VariableV2*
dtype0*
shared_name *
shape:@*
_output_shapes
:@*
	container *!
_class
loc:@v/dense_4/bias
�
v/dense_4/bias/AssignAssignv/dense_4/bias v/dense_4/bias/Initializer/zeros*
use_locking(*
_output_shapes
:@*
T0*
validate_shape(*!
_class
loc:@v/dense_4/bias
w
v/dense_4/bias/readIdentityv/dense_4/bias*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_4/bias
�
v/dense_4/MatMulMatMul	v/Squeezev/dense_4/kernel/read*
transpose_b( *'
_output_shapes
:���������@*
transpose_a( *
T0
�
v/dense_4/BiasAddBiasAddv/dense_4/MatMulv/dense_4/bias/read*
T0*'
_output_shapes
:���������@*
data_formatNHWC
[
v/dense_4/ReluReluv/dense_4/BiasAdd*
T0*'
_output_shapes
:���������@
�
1v/dense_5/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@v/dense_5/kernel*
valueB"@       *
_output_shapes
:*
dtype0
�
/v/dense_5/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *#
_class
loc:@v/dense_5/kernel*
dtype0*
valueB
 *  ��
�
/v/dense_5/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *#
_class
loc:@v/dense_5/kernel*
dtype0*
valueB
 *  �>
�
9v/dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_5/kernel/Initializer/random_uniform/shape*
seed�*#
_class
loc:@v/dense_5/kernel*
seed2�*
dtype0*
T0*
_output_shapes

:@ 
�
/v/dense_5/kernel/Initializer/random_uniform/subSub/v/dense_5/kernel/Initializer/random_uniform/max/v/dense_5/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_5/kernel*
T0*
_output_shapes
: 
�
/v/dense_5/kernel/Initializer/random_uniform/mulMul9v/dense_5/kernel/Initializer/random_uniform/RandomUniform/v/dense_5/kernel/Initializer/random_uniform/sub*#
_class
loc:@v/dense_5/kernel*
T0*
_output_shapes

:@ 
�
+v/dense_5/kernel/Initializer/random_uniformAdd/v/dense_5/kernel/Initializer/random_uniform/mul/v/dense_5/kernel/Initializer/random_uniform/min*
_output_shapes

:@ *
T0*#
_class
loc:@v/dense_5/kernel
�
v/dense_5/kernel
VariableV2*
shape
:@ *#
_class
loc:@v/dense_5/kernel*
	container *
shared_name *
dtype0*
_output_shapes

:@ 
�
v/dense_5/kernel/AssignAssignv/dense_5/kernel+v/dense_5/kernel/Initializer/random_uniform*#
_class
loc:@v/dense_5/kernel*
use_locking(*
_output_shapes

:@ *
validate_shape(*
T0
�
v/dense_5/kernel/readIdentityv/dense_5/kernel*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel*
T0
�
 v/dense_5/bias/Initializer/zerosConst*!
_class
loc:@v/dense_5/bias*
dtype0*
_output_shapes
: *
valueB *    
�
v/dense_5/bias
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
: *
shape: *!
_class
loc:@v/dense_5/bias
�
v/dense_5/bias/AssignAssignv/dense_5/bias v/dense_5/bias/Initializer/zeros*!
_class
loc:@v/dense_5/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
w
v/dense_5/bias/readIdentityv/dense_5/bias*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
T0
�
v/dense_5/MatMulMatMulv/dense_4/Reluv/dense_5/kernel/read*
transpose_b( *
T0*'
_output_shapes
:��������� *
transpose_a( 
�
v/dense_5/BiasAddBiasAddv/dense_5/MatMulv/dense_5/bias/read*'
_output_shapes
:��������� *
T0*
data_formatNHWC
[
v/dense_5/ReluReluv/dense_5/BiasAdd*
T0*'
_output_shapes
:��������� 
�
1v/dense_6/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"       *#
_class
loc:@v/dense_6/kernel*
_output_shapes
:
�
/v/dense_6/kernel/Initializer/random_uniform/minConst*#
_class
loc:@v/dense_6/kernel*
_output_shapes
: *
dtype0*
valueB
 *�Kƾ
�
/v/dense_6/kernel/Initializer/random_uniform/maxConst*
valueB
 *�K�>*
dtype0*
_output_shapes
: *#
_class
loc:@v/dense_6/kernel
�
9v/dense_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_6/kernel/Initializer/random_uniform/shape*
seed2�*
dtype0*
T0*
seed�*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: 
�
/v/dense_6/kernel/Initializer/random_uniform/subSub/v/dense_6/kernel/Initializer/random_uniform/max/v/dense_6/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_6/kernel*
T0*
_output_shapes
: 
�
/v/dense_6/kernel/Initializer/random_uniform/mulMul9v/dense_6/kernel/Initializer/random_uniform/RandomUniform/v/dense_6/kernel/Initializer/random_uniform/sub*#
_class
loc:@v/dense_6/kernel*
T0*
_output_shapes

: 
�
+v/dense_6/kernel/Initializer/random_uniformAdd/v/dense_6/kernel/Initializer/random_uniform/mul/v/dense_6/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
T0
�
v/dense_6/kernel
VariableV2*
shape
: *#
_class
loc:@v/dense_6/kernel*
dtype0*
_output_shapes

: *
	container *
shared_name 
�
v/dense_6/kernel/AssignAssignv/dense_6/kernel+v/dense_6/kernel/Initializer/random_uniform*
_output_shapes

: *
T0*
validate_shape(*#
_class
loc:@v/dense_6/kernel*
use_locking(
�
v/dense_6/kernel/readIdentityv/dense_6/kernel*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
T0
�
 v/dense_6/bias/Initializer/zerosConst*!
_class
loc:@v/dense_6/bias*
valueB*    *
_output_shapes
:*
dtype0
�
v/dense_6/bias
VariableV2*
shared_name *
	container *
shape:*
dtype0*
_output_shapes
:*!
_class
loc:@v/dense_6/bias
�
v/dense_6/bias/AssignAssignv/dense_6/bias v/dense_6/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
T0*
use_locking(
w
v/dense_6/bias/readIdentityv/dense_6/bias*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
T0
�
v/dense_6/MatMulMatMulv/dense_5/Reluv/dense_6/kernel/read*
T0*
transpose_b( *'
_output_shapes
:���������*
transpose_a( 
�
v/dense_6/BiasAddBiasAddv/dense_6/MatMulv/dense_6/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
[
v/dense_6/ReluReluv/dense_6/BiasAdd*'
_output_shapes
:���������*
T0
�
1v/dense_7/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@v/dense_7/kernel*
valueB"      *
_output_shapes
:*
dtype0
�
/v/dense_7/kernel/Initializer/random_uniform/minConst*#
_class
loc:@v/dense_7/kernel*
valueB
 *�Q�*
dtype0*
_output_shapes
: 
�
/v/dense_7/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@v/dense_7/kernel*
dtype0*
valueB
 *�Q?*
_output_shapes
: 
�
9v/dense_7/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_7/kernel/Initializer/random_uniform/shape*
T0*#
_class
loc:@v/dense_7/kernel*
seed2�*
dtype0*
seed�*
_output_shapes

:
�
/v/dense_7/kernel/Initializer/random_uniform/subSub/v/dense_7/kernel/Initializer/random_uniform/max/v/dense_7/kernel/Initializer/random_uniform/min*
_output_shapes
: *#
_class
loc:@v/dense_7/kernel*
T0
�
/v/dense_7/kernel/Initializer/random_uniform/mulMul9v/dense_7/kernel/Initializer/random_uniform/RandomUniform/v/dense_7/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@v/dense_7/kernel*
_output_shapes

:
�
+v/dense_7/kernel/Initializer/random_uniformAdd/v/dense_7/kernel/Initializer/random_uniform/mul/v/dense_7/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_7/kernel*
_output_shapes

:
�
v/dense_7/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name *#
_class
loc:@v/dense_7/kernel
�
v/dense_7/kernel/AssignAssignv/dense_7/kernel+v/dense_7/kernel/Initializer/random_uniform*
_output_shapes

:*
T0*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
use_locking(
�
v/dense_7/kernel/readIdentityv/dense_7/kernel*#
_class
loc:@v/dense_7/kernel*
T0*
_output_shapes

:
�
 v/dense_7/bias/Initializer/zerosConst*!
_class
loc:@v/dense_7/bias*
valueB*    *
dtype0*
_output_shapes
:
�
v/dense_7/bias
VariableV2*
_output_shapes
:*!
_class
loc:@v/dense_7/bias*
shape:*
	container *
shared_name *
dtype0
�
v/dense_7/bias/AssignAssignv/dense_7/bias v/dense_7/bias/Initializer/zeros*!
_class
loc:@v/dense_7/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
w
v/dense_7/bias/readIdentityv/dense_7/bias*
_output_shapes
:*!
_class
loc:@v/dense_7/bias*
T0
�
v/dense_7/MatMulMatMulv/dense_6/Reluv/dense_7/kernel/read*
transpose_a( *
T0*'
_output_shapes
:���������*
transpose_b( 
�
v/dense_7/BiasAddBiasAddv/dense_7/MatMulv/dense_7/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
n
v/Squeeze_1Squeezev/dense_7/BiasAdd*#
_output_shapes
:���������*
squeeze_dims
*
T0
O
subSubpi/SumPlaceholder_5*#
_output_shapes
:���������*
T0
=
ExpExpsub*#
_output_shapes
:���������*
T0
N
	Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
Z
GreaterGreaterPlaceholder_3	Greater/y*
T0*#
_output_shapes
:���������
J
mul/xConst*
_output_shapes
: *
valueB
 *���?*
dtype0
N
mulMulmul/xPlaceholder_3*#
_output_shapes
:���������*
T0
L
mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *��L?
R
mul_1Mulmul_1/xPlaceholder_3*
T0*#
_output_shapes
:���������
S
SelectSelectGreatermulmul_1*#
_output_shapes
:���������*
T0
N
mul_2MulExpPlaceholder_3*
T0*#
_output_shapes
:���������
O
MinimumMinimummul_2Select*
T0*#
_output_shapes
:���������
O
ConstConst*
_output_shapes
:*
valueB: *
dtype0
Z
MeanMeanMinimumConst*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
1
NegNegMean*
_output_shapes
: *
T0
V
sub_1SubPlaceholder_4v/Squeeze_1*
T0*#
_output_shapes
:���������
J
pow/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
F
powPowsub_1pow/y*
T0*#
_output_shapes
:���������
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Z
Mean_1MeanpowConst_1*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
Q
sub_2SubPlaceholder_5pi/Sum*
T0*#
_output_shapes
:���������
Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
\
Mean_2Meansub_2Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
B
Neg_1Negpi/Sum*#
_output_shapes
:���������*
T0
Q
Const_3Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_3MeanNeg_1Const_3*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
P
Greater_1/yConst*
valueB
 *���?*
dtype0*
_output_shapes
: 
T
	Greater_1GreaterExpGreater_1/y*
T0*#
_output_shapes
:���������
K
Less/yConst*
valueB
 *��L?*
_output_shapes
: *
dtype0
G
LessLessExpLess/y*#
_output_shapes
:���������*
T0
L
	LogicalOr	LogicalOr	Greater_1Less*#
_output_shapes
:���������
d
CastCast	LogicalOr*
Truncate( *#
_output_shapes
:���������*

SrcT0
*

DstT0
Q
Const_4Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_4MeanCastConst_4*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
R
gradients/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
X
gradients/grad_ys_0Const*
_output_shapes
: *
dtype0*
valueB
 *  �?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*

index_type0*
_output_shapes
: *
T0
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
Tshape0*
T0*
_output_shapes
:
`
gradients/Mean_grad/ShapeShapeMinimum*
out_type0*
_output_shapes
:*
T0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
b
gradients/Mean_grad/Shape_1ShapeMinimum*
T0*
_output_shapes
:*
out_type0
^
gradients/Mean_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
T0*

Tidx0*
	keep_dims( 
_
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
_output_shapes
: *

DstT0*
Truncate( *

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
a
gradients/Minimum_grad/ShapeShapemul_2*
_output_shapes
:*
out_type0*
T0
d
gradients/Minimum_grad/Shape_1ShapeSelect*
out_type0*
T0*
_output_shapes
:
y
gradients/Minimum_grad/Shape_2Shapegradients/Mean_grad/truediv*
T0*
out_type0*
_output_shapes
:
g
"gradients/Minimum_grad/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*

index_type0*
T0*#
_output_shapes
:���������
j
 gradients/Minimum_grad/LessEqual	LessEqualmul_2Select*#
_output_shapes
:���������*
T0
�
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/Mean_grad/truedivgradients/Minimum_grad/zeros*#
_output_shapes
:���������*
T0
�
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
�
gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
�
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/Mean_grad/truediv*#
_output_shapes
:���������*
T0
�
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
�
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*1
_class'
%#loc:@gradients/Minimum_grad/Reshape*#
_output_shapes
:���������*
T0
�
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1*#
_output_shapes
:���������*
T0
]
gradients/mul_2_grad/ShapeShapeExp*
_output_shapes
:*
T0*
out_type0
i
gradients/mul_2_grad/Shape_1ShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
�
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependencyPlaceholder_3*#
_output_shapes
:���������*
T0
�
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
�
gradients/mul_2_grad/Mul_1MulExp/gradients/Minimum_grad/tuple/control_dependency*#
_output_shapes
:���������*
T0
�
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:���������
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
�
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
�
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*
T0*#
_output_shapes
:���������

gradients/Exp_grad/mulMul-gradients/mul_2_grad/tuple/control_dependencyExp*#
_output_shapes
:���������*
T0
^
gradients/sub_grad/ShapeShapepi/Sum*
_output_shapes
:*
out_type0*
T0
g
gradients/sub_grad/Shape_1ShapePlaceholder_5*
T0*
_output_shapes
:*
out_type0
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*
T0*#
_output_shapes
:���������
�
gradients/sub_grad/Sum_1Sumgradients/Exp_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*-
_class#
!loc:@gradients/sub_grad/Reshape*#
_output_shapes
:���������*
T0
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*#
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
c
gradients/pi/Sum_grad/ShapeShapepi/mul_1*
out_type0*
_output_shapes
:*
T0
�
gradients/pi/Sum_grad/SizeConst*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
dtype0
�
gradients/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
T0
�
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0
�
gradients/pi/Sum_grad/Shape_1Const*
valueB *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
dtype0
�
!gradients/pi/Sum_grad/range/startConst*
dtype0*
value	B : *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
�
!gradients/pi/Sum_grad/range/deltaConst*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B :*
_output_shapes
: 
�
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*

Tidx0*
_output_shapes
:
�
 gradients/pi/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B :
�
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*

index_type0*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
�
#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:*
T0*
N
�
gradients/pi/Sum_grad/Maximum/yConst*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
�
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*
_output_shapes
:*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
�
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:
�
gradients/pi/Sum_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency#gradients/pi/Sum_grad/DynamicStitch*
Tshape0*
T0*0
_output_shapes
:������������������
�
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*(
_output_shapes
:����������*
T0*

Tmultiples0
g
gradients/pi/mul_1_grad/ShapeShape
pi/one_hot*
_output_shapes
:*
out_type0*
T0
l
gradients/pi/mul_1_grad/Shape_1Shapepi/LogSoftmax*
_output_shapes
:*
out_type0*
T0
�
-gradients/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_1_grad/Shapegradients/pi/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/pi/mul_1_grad/MulMulgradients/pi/Sum_grad/Tilepi/LogSoftmax*(
_output_shapes
:����������*
T0
�
gradients/pi/mul_1_grad/SumSumgradients/pi/mul_1_grad/Mul-gradients/pi/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/pi/mul_1_grad/ReshapeReshapegradients/pi/mul_1_grad/Sumgradients/pi/mul_1_grad/Shape*
Tshape0*(
_output_shapes
:����������*
T0

gradients/pi/mul_1_grad/Mul_1Mul
pi/one_hotgradients/pi/Sum_grad/Tile*
T0*(
_output_shapes
:����������
�
gradients/pi/mul_1_grad/Sum_1Sumgradients/pi/mul_1_grad/Mul_1/gradients/pi/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
�
!gradients/pi/mul_1_grad/Reshape_1Reshapegradients/pi/mul_1_grad/Sum_1gradients/pi/mul_1_grad/Shape_1*(
_output_shapes
:����������*
Tshape0*
T0
v
(gradients/pi/mul_1_grad/tuple/group_depsNoOp ^gradients/pi/mul_1_grad/Reshape"^gradients/pi/mul_1_grad/Reshape_1
�
0gradients/pi/mul_1_grad/tuple/control_dependencyIdentitygradients/pi/mul_1_grad/Reshape)^gradients/pi/mul_1_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*2
_class(
&$loc:@gradients/pi/mul_1_grad/Reshape
�
2gradients/pi/mul_1_grad/tuple/control_dependency_1Identity!gradients/pi/mul_1_grad/Reshape_1)^gradients/pi/mul_1_grad/tuple/group_deps*4
_class*
(&loc:@gradients/pi/mul_1_grad/Reshape_1*
T0*(
_output_shapes
:����������
i
 gradients/pi/LogSoftmax_grad/ExpExppi/LogSoftmax*
T0*(
_output_shapes
:����������
}
2gradients/pi/LogSoftmax_grad/Sum/reduction_indicesConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
 gradients/pi/LogSoftmax_grad/SumSum2gradients/pi/mul_1_grad/tuple/control_dependency_12gradients/pi/LogSoftmax_grad/Sum/reduction_indices*
T0*'
_output_shapes
:���������*

Tidx0*
	keep_dims(
�
 gradients/pi/LogSoftmax_grad/mulMul gradients/pi/LogSoftmax_grad/Sum gradients/pi/LogSoftmax_grad/Exp*(
_output_shapes
:����������*
T0
�
 gradients/pi/LogSoftmax_grad/subSub2gradients/pi/mul_1_grad/tuple/control_dependency_1 gradients/pi/LogSoftmax_grad/mul*(
_output_shapes
:����������*
T0
e
gradients/pi/add_grad/ShapeShape
pi/Squeeze*
T0*
_output_shapes
:*
out_type0
c
gradients/pi/add_grad/Shape_1Shapepi/mul*
T0*
out_type0*
_output_shapes
:
�
+gradients/pi/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_grad/Shapegradients/pi/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/pi/add_grad/SumSum gradients/pi/LogSoftmax_grad/sub+gradients/pi/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
gradients/pi/add_grad/ReshapeReshapegradients/pi/add_grad/Sumgradients/pi/add_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/pi/add_grad/Sum_1Sum gradients/pi/LogSoftmax_grad/sub-gradients/pi/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
gradients/pi/add_grad/Reshape_1Reshapegradients/pi/add_grad/Sum_1gradients/pi/add_grad/Shape_1*
Tshape0*(
_output_shapes
:����������*
T0
p
&gradients/pi/add_grad/tuple/group_depsNoOp^gradients/pi/add_grad/Reshape ^gradients/pi/add_grad/Reshape_1
�
.gradients/pi/add_grad/tuple/control_dependencyIdentitygradients/pi/add_grad/Reshape'^gradients/pi/add_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/pi/add_grad/Reshape*(
_output_shapes
:����������
�
0gradients/pi/add_grad/tuple/control_dependency_1Identitygradients/pi/add_grad/Reshape_1'^gradients/pi/add_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*2
_class(
&$loc:@gradients/pi/add_grad/Reshape_1
q
gradients/pi/Squeeze_grad/ShapeShapepi/dense_3/BiasAdd*
T0*
out_type0*
_output_shapes
:
�
!gradients/pi/Squeeze_grad/ReshapeReshape.gradients/pi/add_grad/tuple/control_dependencygradients/pi/Squeeze_grad/Shape*,
_output_shapes
:����������*
Tshape0*
T0
�
-gradients/pi/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad!gradients/pi/Squeeze_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
�
2gradients/pi/dense_3/BiasAdd_grad/tuple/group_depsNoOp"^gradients/pi/Squeeze_grad/Reshape.^gradients/pi/dense_3/BiasAdd_grad/BiasAddGrad
�
:gradients/pi/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity!gradients/pi/Squeeze_grad/Reshape3^gradients/pi/dense_3/BiasAdd_grad/tuple/group_deps*4
_class*
(&loc:@gradients/pi/Squeeze_grad/Reshape*
T0*,
_output_shapes
:����������
�
<gradients/pi/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_3/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_3/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*@
_class6
42loc:@gradients/pi/dense_3/BiasAdd_grad/BiasAddGrad*
T0
�
)gradients/pi/dense_3/Tensordot_grad/ShapeShapepi/dense_3/Tensordot/MatMul*
_output_shapes
:*
T0*
out_type0
�
+gradients/pi/dense_3/Tensordot_grad/ReshapeReshape:gradients/pi/dense_3/BiasAdd_grad/tuple/control_dependency)gradients/pi/dense_3/Tensordot_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
1gradients/pi/dense_3/Tensordot/MatMul_grad/MatMulMatMul+gradients/pi/dense_3/Tensordot_grad/Reshapepi/dense_3/Tensordot/Reshape_1*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:���������
�
3gradients/pi/dense_3/Tensordot/MatMul_grad/MatMul_1MatMulpi/dense_3/Tensordot/Reshape+gradients/pi/dense_3/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:���������
�
;gradients/pi/dense_3/Tensordot/MatMul_grad/tuple/group_depsNoOp2^gradients/pi/dense_3/Tensordot/MatMul_grad/MatMul4^gradients/pi/dense_3/Tensordot/MatMul_grad/MatMul_1
�
Cgradients/pi/dense_3/Tensordot/MatMul_grad/tuple/control_dependencyIdentity1gradients/pi/dense_3/Tensordot/MatMul_grad/MatMul<^gradients/pi/dense_3/Tensordot/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*D
_class:
86loc:@gradients/pi/dense_3/Tensordot/MatMul_grad/MatMul
�
Egradients/pi/dense_3/Tensordot/MatMul_grad/tuple/control_dependency_1Identity3gradients/pi/dense_3/Tensordot/MatMul_grad/MatMul_1<^gradients/pi/dense_3/Tensordot/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*F
_class<
:8loc:@gradients/pi/dense_3/Tensordot/MatMul_grad/MatMul_1
�
1gradients/pi/dense_3/Tensordot/Reshape_grad/ShapeShapepi/dense_3/Tensordot/transpose*
out_type0*
T0*
_output_shapes
:
�
3gradients/pi/dense_3/Tensordot/Reshape_grad/ReshapeReshapeCgradients/pi/dense_3/Tensordot/MatMul_grad/tuple/control_dependency1gradients/pi/dense_3/Tensordot/Reshape_grad/Shape*,
_output_shapes
:����������*
Tshape0*
T0
�
3gradients/pi/dense_3/Tensordot/Reshape_1_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
5gradients/pi/dense_3/Tensordot/Reshape_1_grad/ReshapeReshapeEgradients/pi/dense_3/Tensordot/MatMul_grad/tuple/control_dependency_13gradients/pi/dense_3/Tensordot/Reshape_1_grad/Shape*
T0*
_output_shapes

:*
Tshape0
�
?gradients/pi/dense_3/Tensordot/transpose_grad/InvertPermutationInvertPermutationpi/dense_3/Tensordot/concat*
T0*
_output_shapes
:
�
7gradients/pi/dense_3/Tensordot/transpose_grad/transpose	Transpose3gradients/pi/dense_3/Tensordot/Reshape_grad/Reshape?gradients/pi/dense_3/Tensordot/transpose_grad/InvertPermutation*
T0*,
_output_shapes
:����������*
Tperm0
�
Agradients/pi/dense_3/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation%pi/dense_3/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
9gradients/pi/dense_3/Tensordot/transpose_1_grad/transpose	Transpose5gradients/pi/dense_3/Tensordot/Reshape_1_grad/ReshapeAgradients/pi/dense_3/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
_output_shapes

:*
T0
�
'gradients/pi/dense_2/Relu_grad/ReluGradReluGrad7gradients/pi/dense_3/Tensordot/transpose_grad/transposepi/dense_2/Relu*,
_output_shapes
:����������*
T0
�
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_2/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
�
2gradients/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad(^gradients/pi/dense_2/Relu_grad/ReluGrad
�
:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi/dense_2/Relu_grad/ReluGrad3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense_2/Relu_grad/ReluGrad*,
_output_shapes
:����������
�
<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
)gradients/pi/dense_2/Tensordot_grad/ShapeShapepi/dense_2/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
�
+gradients/pi/dense_2/Tensordot_grad/ReshapeReshape:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency)gradients/pi/dense_2/Tensordot_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
1gradients/pi/dense_2/Tensordot/MatMul_grad/MatMulMatMul+gradients/pi/dense_2/Tensordot_grad/Reshapepi/dense_2/Tensordot/Reshape_1*'
_output_shapes
:���������*
transpose_a( *
T0*
transpose_b(
�
3gradients/pi/dense_2/Tensordot/MatMul_grad/MatMul_1MatMulpi/dense_2/Tensordot/Reshape+gradients/pi/dense_2/Tensordot_grad/Reshape*
transpose_a(*
transpose_b( *
T0*'
_output_shapes
:���������
�
;gradients/pi/dense_2/Tensordot/MatMul_grad/tuple/group_depsNoOp2^gradients/pi/dense_2/Tensordot/MatMul_grad/MatMul4^gradients/pi/dense_2/Tensordot/MatMul_grad/MatMul_1
�
Cgradients/pi/dense_2/Tensordot/MatMul_grad/tuple/control_dependencyIdentity1gradients/pi/dense_2/Tensordot/MatMul_grad/MatMul<^gradients/pi/dense_2/Tensordot/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*D
_class:
86loc:@gradients/pi/dense_2/Tensordot/MatMul_grad/MatMul
�
Egradients/pi/dense_2/Tensordot/MatMul_grad/tuple/control_dependency_1Identity3gradients/pi/dense_2/Tensordot/MatMul_grad/MatMul_1<^gradients/pi/dense_2/Tensordot/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/pi/dense_2/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:
�
1gradients/pi/dense_2/Tensordot/Reshape_grad/ShapeShapepi/dense_2/Tensordot/transpose*
_output_shapes
:*
T0*
out_type0
�
3gradients/pi/dense_2/Tensordot/Reshape_grad/ReshapeReshapeCgradients/pi/dense_2/Tensordot/MatMul_grad/tuple/control_dependency1gradients/pi/dense_2/Tensordot/Reshape_grad/Shape*
T0*,
_output_shapes
:����������*
Tshape0
�
3gradients/pi/dense_2/Tensordot/Reshape_1_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
5gradients/pi/dense_2/Tensordot/Reshape_1_grad/ReshapeReshapeEgradients/pi/dense_2/Tensordot/MatMul_grad/tuple/control_dependency_13gradients/pi/dense_2/Tensordot/Reshape_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
�
?gradients/pi/dense_2/Tensordot/transpose_grad/InvertPermutationInvertPermutationpi/dense_2/Tensordot/concat*
_output_shapes
:*
T0
�
7gradients/pi/dense_2/Tensordot/transpose_grad/transpose	Transpose3gradients/pi/dense_2/Tensordot/Reshape_grad/Reshape?gradients/pi/dense_2/Tensordot/transpose_grad/InvertPermutation*
T0*
Tperm0*,
_output_shapes
:����������
�
Agradients/pi/dense_2/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation%pi/dense_2/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
�
9gradients/pi/dense_2/Tensordot/transpose_1_grad/transpose	Transpose5gradients/pi/dense_2/Tensordot/Reshape_1_grad/ReshapeAgradients/pi/dense_2/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:
�
'gradients/pi/dense_1/Relu_grad/ReluGradReluGrad7gradients/pi/dense_2/Tensordot/transpose_grad/transposepi/dense_1/Relu*
T0*,
_output_shapes
:����������
�
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Relu_grad/ReluGrad*
_output_shapes
:*
T0*
data_formatNHWC
�
2gradients/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad(^gradients/pi/dense_1/Relu_grad/ReluGrad
�
:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/Relu_grad/ReluGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense_1/Relu_grad/ReluGrad*,
_output_shapes
:����������
�
<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
)gradients/pi/dense_1/Tensordot_grad/ShapeShapepi/dense_1/Tensordot/MatMul*
_output_shapes
:*
T0*
out_type0
�
+gradients/pi/dense_1/Tensordot_grad/ReshapeReshape:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency)gradients/pi/dense_1/Tensordot_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
�
1gradients/pi/dense_1/Tensordot/MatMul_grad/MatMulMatMul+gradients/pi/dense_1/Tensordot_grad/Reshapepi/dense_1/Tensordot/Reshape_1*'
_output_shapes
:��������� *
transpose_a( *
transpose_b(*
T0
�
3gradients/pi/dense_1/Tensordot/MatMul_grad/MatMul_1MatMulpi/dense_1/Tensordot/Reshape+gradients/pi/dense_1/Tensordot_grad/Reshape*
T0*
transpose_b( *
transpose_a(*'
_output_shapes
:���������
�
;gradients/pi/dense_1/Tensordot/MatMul_grad/tuple/group_depsNoOp2^gradients/pi/dense_1/Tensordot/MatMul_grad/MatMul4^gradients/pi/dense_1/Tensordot/MatMul_grad/MatMul_1
�
Cgradients/pi/dense_1/Tensordot/MatMul_grad/tuple/control_dependencyIdentity1gradients/pi/dense_1/Tensordot/MatMul_grad/MatMul<^gradients/pi/dense_1/Tensordot/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/pi/dense_1/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:��������� 
�
Egradients/pi/dense_1/Tensordot/MatMul_grad/tuple/control_dependency_1Identity3gradients/pi/dense_1/Tensordot/MatMul_grad/MatMul_1<^gradients/pi/dense_1/Tensordot/MatMul_grad/tuple/group_deps*
_output_shapes

: *F
_class<
:8loc:@gradients/pi/dense_1/Tensordot/MatMul_grad/MatMul_1*
T0
�
1gradients/pi/dense_1/Tensordot/Reshape_grad/ShapeShapepi/dense_1/Tensordot/transpose*
T0*
_output_shapes
:*
out_type0
�
3gradients/pi/dense_1/Tensordot/Reshape_grad/ReshapeReshapeCgradients/pi/dense_1/Tensordot/MatMul_grad/tuple/control_dependency1gradients/pi/dense_1/Tensordot/Reshape_grad/Shape*
Tshape0*
T0*,
_output_shapes
:���������� 
�
3gradients/pi/dense_1/Tensordot/Reshape_1_grad/ShapeConst*
dtype0*
valueB"       *
_output_shapes
:
�
5gradients/pi/dense_1/Tensordot/Reshape_1_grad/ReshapeReshapeEgradients/pi/dense_1/Tensordot/MatMul_grad/tuple/control_dependency_13gradients/pi/dense_1/Tensordot/Reshape_1_grad/Shape*
_output_shapes

: *
T0*
Tshape0
�
?gradients/pi/dense_1/Tensordot/transpose_grad/InvertPermutationInvertPermutationpi/dense_1/Tensordot/concat*
_output_shapes
:*
T0
�
7gradients/pi/dense_1/Tensordot/transpose_grad/transpose	Transpose3gradients/pi/dense_1/Tensordot/Reshape_grad/Reshape?gradients/pi/dense_1/Tensordot/transpose_grad/InvertPermutation*
Tperm0*,
_output_shapes
:���������� *
T0
�
Agradients/pi/dense_1/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation%pi/dense_1/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
9gradients/pi/dense_1/Tensordot/transpose_1_grad/transpose	Transpose5gradients/pi/dense_1/Tensordot/Reshape_1_grad/ReshapeAgradients/pi/dense_1/Tensordot/transpose_1_grad/InvertPermutation*
_output_shapes

: *
Tperm0*
T0
�
%gradients/pi/dense/Relu_grad/ReluGradReluGrad7gradients/pi/dense_1/Tensordot/transpose_grad/transposepi/dense/Relu*,
_output_shapes
:���������� *
T0
�
+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
�
0gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp,^gradients/pi/dense/BiasAdd_grad/BiasAddGrad&^gradients/pi/dense/Relu_grad/ReluGrad
�
8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/pi/dense/Relu_grad/ReluGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*8
_class.
,*loc:@gradients/pi/dense/Relu_grad/ReluGrad*
T0*,
_output_shapes
:���������� 
�
:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/pi/dense/BiasAdd_grad/BiasAddGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*>
_class4
20loc:@gradients/pi/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
�
'gradients/pi/dense/Tensordot_grad/ShapeShapepi/dense/Tensordot/MatMul*
_output_shapes
:*
T0*
out_type0
�
)gradients/pi/dense/Tensordot_grad/ReshapeReshape8gradients/pi/dense/BiasAdd_grad/tuple/control_dependency'gradients/pi/dense/Tensordot_grad/Shape*
Tshape0*
T0*'
_output_shapes
:��������� 
�
/gradients/pi/dense/Tensordot/MatMul_grad/MatMulMatMul)gradients/pi/dense/Tensordot_grad/Reshapepi/dense/Tensordot/Reshape_1*
transpose_a( *
transpose_b(*'
_output_shapes
:���������*
T0
�
1gradients/pi/dense/Tensordot/MatMul_grad/MatMul_1MatMulpi/dense/Tensordot/Reshape)gradients/pi/dense/Tensordot_grad/Reshape*
T0*'
_output_shapes
:��������� *
transpose_a(*
transpose_b( 
�
9gradients/pi/dense/Tensordot/MatMul_grad/tuple/group_depsNoOp0^gradients/pi/dense/Tensordot/MatMul_grad/MatMul2^gradients/pi/dense/Tensordot/MatMul_grad/MatMul_1
�
Agradients/pi/dense/Tensordot/MatMul_grad/tuple/control_dependencyIdentity/gradients/pi/dense/Tensordot/MatMul_grad/MatMul:^gradients/pi/dense/Tensordot/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/pi/dense/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Cgradients/pi/dense/Tensordot/MatMul_grad/tuple/control_dependency_1Identity1gradients/pi/dense/Tensordot/MatMul_grad/MatMul_1:^gradients/pi/dense/Tensordot/MatMul_grad/tuple/group_deps*
_output_shapes

: *
T0*D
_class:
86loc:@gradients/pi/dense/Tensordot/MatMul_grad/MatMul_1
�
1gradients/pi/dense/Tensordot/Reshape_1_grad/ShapeConst*
dtype0*
valueB"       *
_output_shapes
:
�
3gradients/pi/dense/Tensordot/Reshape_1_grad/ReshapeReshapeCgradients/pi/dense/Tensordot/MatMul_grad/tuple/control_dependency_11gradients/pi/dense/Tensordot/Reshape_1_grad/Shape*
Tshape0*
T0*
_output_shapes

: 
�
?gradients/pi/dense/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation#pi/dense/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
7gradients/pi/dense/Tensordot/transpose_1_grad/transpose	Transpose3gradients/pi/dense/Tensordot/Reshape_1_grad/Reshape?gradients/pi/dense/Tensordot/transpose_1_grad/InvertPermutation*
_output_shapes

: *
Tperm0*
T0
�
beta1_power/initial_valueConst*
valueB
 *fff?* 
_class
loc:@pi/dense/bias*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
_output_shapes
: *
shared_name *
dtype0* 
_class
loc:@pi/dense/bias*
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
l
beta1_power/readIdentitybeta1_power* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0
�
beta2_power/initial_valueConst*
valueB
 *w�?*
dtype0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
beta2_power
VariableV2*
dtype0* 
_class
loc:@pi/dense/bias*
shared_name *
shape: *
_output_shapes
: *
	container 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_output_shapes
: *
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias
l
beta2_power/readIdentitybeta2_power*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0
�
&pi/dense/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
valueB *    
�
pi/dense/kernel/Adam
VariableV2*
_output_shapes

: *
shared_name *
	container *"
_class
loc:@pi/dense/kernel*
shape
: *
dtype0
�
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*
_output_shapes

: *"
_class
loc:@pi/dense/kernel
�
pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

: 
�
(pi/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
valueB *    *
dtype0
�
pi/dense/kernel/Adam_1
VariableV2*
_output_shapes

: *
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
dtype0*
shape
: 
�
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes

: *
T0*"
_class
loc:@pi/dense/kernel
�
pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

: 
�
$pi/dense/bias/Adam/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes
: * 
_class
loc:@pi/dense/bias
�
pi/dense/bias/Adam
VariableV2* 
_class
loc:@pi/dense/bias*
shape: *
shared_name *
	container *
_output_shapes
: *
dtype0
�
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
~
pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
�
&pi/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
valueB *    
�
pi/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape: 
�
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
: 
�
pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
�
(pi/dense_1/kernel/Adam/Initializer/zerosConst*
dtype0*
valueB *    *$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: 
�
pi/dense_1/kernel/Adam
VariableV2*
shared_name *
	container *$
_class
loc:@pi/dense_1/kernel*
shape
: *
dtype0*
_output_shapes

: 
�
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*$
_class
loc:@pi/dense_1/kernel
�
pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: 
�
*pi/dense_1/kernel/Adam_1/Initializer/zerosConst*
dtype0*
valueB *    *$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: 
�
pi/dense_1/kernel/Adam_1
VariableV2*
	container *$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes

: *
shared_name *
shape
: 
�
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel*
validate_shape(
�
pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

: 
�
&pi/dense_1/bias/Adam/Initializer/zerosConst*
valueB*    *"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
dtype0
�
pi/dense_1/bias/Adam
VariableV2*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
	container *
dtype0*
shape:*
shared_name 
�
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:
�
pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_1/bias
�
(pi/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
dtype0*
valueB*    *
_output_shapes
:
�
pi/dense_1/bias/Adam_1
VariableV2*
shape:*
_output_shapes
:*
shared_name *
	container *"
_class
loc:@pi/dense_1/bias*
dtype0
�
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(
�
pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*
_output_shapes
:*"
_class
loc:@pi/dense_1/bias*
T0
�
(pi/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
valueB*    
�
pi/dense_2/kernel/Adam
VariableV2*$
_class
loc:@pi/dense_2/kernel*
	container *
_output_shapes

:*
shape
:*
dtype0*
shared_name 
�
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:
�
pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*
_output_shapes

:*
T0*$
_class
loc:@pi/dense_2/kernel
�
*pi/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:*
valueB*    
�
pi/dense_2/kernel/Adam_1
VariableV2*$
_class
loc:@pi/dense_2/kernel*
dtype0*
	container *
shape
:*
_output_shapes

:*
shared_name 
�
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*
_output_shapes

:*
T0*$
_class
loc:@pi/dense_2/kernel
�
&pi/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*"
_class
loc:@pi/dense_2/bias*
valueB*    *
_output_shapes
:
�
pi/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
dtype0*
shape:*
shared_name *
	container *"
_class
loc:@pi/dense_2/bias
�
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(
�
pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
�
(pi/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *"
_class
loc:@pi/dense_2/bias
�
pi/dense_2/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@pi/dense_2/bias
�
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
�
pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
�
(pi/dense_3/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*
dtype0*$
_class
loc:@pi/dense_3/kernel*
valueB*    
�
pi/dense_3/kernel/Adam
VariableV2*
shape
:*
	container *
dtype0*
shared_name *$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:
�
pi/dense_3/kernel/Adam/AssignAssignpi/dense_3/kernel/Adam(pi/dense_3/kernel/Adam/Initializer/zeros*
T0*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
validate_shape(*
use_locking(
�
pi/dense_3/kernel/Adam/readIdentitypi/dense_3/kernel/Adam*
T0*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:
�
*pi/dense_3/kernel/Adam_1/Initializer/zerosConst*
dtype0*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
valueB*    
�
pi/dense_3/kernel/Adam_1
VariableV2*$
_class
loc:@pi/dense_3/kernel*
dtype0*
shared_name *
shape
:*
	container *
_output_shapes

:
�
pi/dense_3/kernel/Adam_1/AssignAssignpi/dense_3/kernel/Adam_1*pi/dense_3/kernel/Adam_1/Initializer/zeros*
T0*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
_output_shapes

:*
validate_shape(
�
pi/dense_3/kernel/Adam_1/readIdentitypi/dense_3/kernel/Adam_1*
_output_shapes

:*$
_class
loc:@pi/dense_3/kernel*
T0
�
&pi/dense_3/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:*
valueB*    *
dtype0
�
pi/dense_3/bias/Adam
VariableV2*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
	container *
shared_name *
dtype0*
shape:
�
pi/dense_3/bias/Adam/AssignAssignpi/dense_3/bias/Adam&pi/dense_3/bias/Adam/Initializer/zeros*
use_locking(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_3/bias*
validate_shape(
�
pi/dense_3/bias/Adam/readIdentitypi/dense_3/bias/Adam*"
_class
loc:@pi/dense_3/bias*
T0*
_output_shapes
:
�
(pi/dense_3/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*
valueB*    *"
_class
loc:@pi/dense_3/bias
�
pi/dense_3/bias/Adam_1
VariableV2*
_output_shapes
:*
shared_name *
shape:*
dtype0*
	container *"
_class
loc:@pi/dense_3/bias
�
pi/dense_3/bias/Adam_1/AssignAssignpi/dense_3/bias/Adam_1(pi/dense_3/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
T0*
validate_shape(
�
pi/dense_3/bias/Adam_1/readIdentitypi/dense_3/bias/Adam_1*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:*
T0
W
Adam/learning_rateConst*
valueB
 *RI�9*
_output_shapes
: *
dtype0
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
_output_shapes
: *
dtype0*
valueB
 *w�?
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w�+2
�
%Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/pi/dense/Tensordot/transpose_1_grad/transpose*
_output_shapes

: *
T0*
use_nesterov( *
use_locking( *"
_class
loc:@pi/dense/kernel
�
#Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_nesterov( * 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking( 
�
'Adam/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/pi/dense_1/Tensordot/transpose_1_grad/transpose*
use_nesterov( *$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: *
use_locking( *
T0
�
%Adam/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1*"
_class
loc:@pi/dense_1/bias*
T0*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
'Adam/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/pi/dense_2/Tensordot/transpose_1_grad/transpose*
T0*
use_nesterov( *
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
use_locking( 
�
%Adam/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking( *
use_nesterov( 
�
'Adam/update_pi/dense_3/kernel/ApplyAdam	ApplyAdampi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/pi/dense_3/Tensordot/transpose_1_grad/transpose*
T0*
_output_shapes

:*
use_locking( *$
_class
loc:@pi/dense_3/kernel*
use_nesterov( 
�
%Adam/update_pi/dense_3/bias/ApplyAdam	ApplyAdampi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/pi/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam&^Adam/update_pi/dense_3/bias/ApplyAdam(^Adam/update_pi/dense_3/kernel/ApplyAdam* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam&^Adam/update_pi/dense_3/bias/ApplyAdam(^Adam/update_pi/dense_3/kernel/ApplyAdam*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking( 
�
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam&^Adam/update_pi/dense_3/bias/ApplyAdam(^Adam/update_pi/dense_3/kernel/ApplyAdam
T
gradients_1/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_1/grad_ys_0Const*
valueB
 *  �?*
_output_shapes
: *
dtype0
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
�
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients_1/Mean_1_grad/ShapeShapepow*
_output_shapes
:*
T0*
out_type0
�
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
_output_shapes
: *
dtype0
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_1_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
�
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
_output_shapes
: *
T0
�
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
Truncate( *

DstT0*

SrcT0*
_output_shapes
: 
�
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*#
_output_shapes
:���������*
T0
_
gradients_1/pow_grad/ShapeShapesub_1*
out_type0*
T0*
_output_shapes
:
_
gradients_1/pow_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
�
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*
T0*#
_output_shapes
:���������
_
gradients_1/pow_grad/sub/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
T0*
_output_shapes
: 
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*#
_output_shapes
:���������*
T0
�
gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*
T0*#
_output_shapes
:���������
�
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
�
gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
c
gradients_1/pow_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*
T0*#
_output_shapes
:���������
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
T0*
_output_shapes
:*
out_type0
i
$gradients_1/pow_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*#
_output_shapes
:���������*

index_type0*
T0
�
gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*
T0*#
_output_shapes
:���������
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*#
_output_shapes
:���������*
T0
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*#
_output_shapes
:���������*
T0
�
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*#
_output_shapes
:���������*
T0
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*
T0*#
_output_shapes
:���������
�
gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*#
_output_shapes
:���������*
T0
�
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
�
gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
�
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*/
_class%
#!loc:@gradients_1/pow_grad/Reshape
�
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1*
_output_shapes
: *
T0
i
gradients_1/sub_1_grad/ShapeShapePlaceholder_4*
_output_shapes
:*
T0*
out_type0
i
gradients_1/sub_1_grad/Shape_1Shapev/Squeeze_1*
T0*
_output_shapes
:*
out_type0
�
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
Tshape0*#
_output_shapes
:���������*
T0
�
gradients_1/sub_1_grad/Sum_1Sum-gradients_1/pow_grad/tuple/control_dependency.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
b
gradients_1/sub_1_grad/NegNeggradients_1/sub_1_grad/Sum_1*
_output_shapes
:*
T0
�
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Neggradients_1/sub_1_grad/Shape_1*#
_output_shapes
:���������*
Tshape0*
T0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
�
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*
T0*#
_output_shapes
:���������
�
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*
T0*#
_output_shapes
:���������
s
"gradients_1/v/Squeeze_1_grad/ShapeShapev/dense_7/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
$gradients_1/v/Squeeze_1_grad/ReshapeReshape1gradients_1/sub_1_grad/tuple/control_dependency_1"gradients_1/v/Squeeze_1_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
.gradients_1/v/dense_7/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients_1/v/Squeeze_1_grad/Reshape*
T0*
_output_shapes
:*
data_formatNHWC
�
3gradients_1/v/dense_7/BiasAdd_grad/tuple/group_depsNoOp%^gradients_1/v/Squeeze_1_grad/Reshape/^gradients_1/v/dense_7/BiasAdd_grad/BiasAddGrad
�
;gradients_1/v/dense_7/BiasAdd_grad/tuple/control_dependencyIdentity$gradients_1/v/Squeeze_1_grad/Reshape4^gradients_1/v/dense_7/BiasAdd_grad/tuple/group_deps*7
_class-
+)loc:@gradients_1/v/Squeeze_1_grad/Reshape*
T0*'
_output_shapes
:���������
�
=gradients_1/v/dense_7/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_7/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_7/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*A
_class7
53loc:@gradients_1/v/dense_7/BiasAdd_grad/BiasAddGrad
�
(gradients_1/v/dense_7/MatMul_grad/MatMulMatMul;gradients_1/v/dense_7/BiasAdd_grad/tuple/control_dependencyv/dense_7/kernel/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
*gradients_1/v/dense_7/MatMul_grad/MatMul_1MatMulv/dense_6/Relu;gradients_1/v/dense_7/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:*
T0*
transpose_a(
�
2gradients_1/v/dense_7/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_7/MatMul_grad/MatMul+^gradients_1/v/dense_7/MatMul_grad/MatMul_1
�
:gradients_1/v/dense_7/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_7/MatMul_grad/MatMul3^gradients_1/v/dense_7/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*;
_class1
/-loc:@gradients_1/v/dense_7/MatMul_grad/MatMul
�
<gradients_1/v/dense_7/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_7/MatMul_grad/MatMul_13^gradients_1/v/dense_7/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients_1/v/dense_7/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
�
(gradients_1/v/dense_6/Relu_grad/ReluGradReluGrad:gradients_1/v/dense_7/MatMul_grad/tuple/control_dependencyv/dense_6/Relu*'
_output_shapes
:���������*
T0
�
.gradients_1/v/dense_6/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_6/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
�
3gradients_1/v/dense_6/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_6/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_6/Relu_grad/ReluGrad
�
;gradients_1/v/dense_6/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_6/Relu_grad/ReluGrad4^gradients_1/v/dense_6/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@gradients_1/v/dense_6/Relu_grad/ReluGrad
�
=gradients_1/v/dense_6/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_6/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_6/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/v/dense_6/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
(gradients_1/v/dense_6/MatMul_grad/MatMulMatMul;gradients_1/v/dense_6/BiasAdd_grad/tuple/control_dependencyv/dense_6/kernel/read*'
_output_shapes
:��������� *
transpose_a( *
T0*
transpose_b(
�
*gradients_1/v/dense_6/MatMul_grad/MatMul_1MatMulv/dense_5/Relu;gradients_1/v/dense_6/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(*
_output_shapes

: 
�
2gradients_1/v/dense_6/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_6/MatMul_grad/MatMul+^gradients_1/v/dense_6/MatMul_grad/MatMul_1
�
:gradients_1/v/dense_6/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_6/MatMul_grad/MatMul3^gradients_1/v/dense_6/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense_6/MatMul_grad/MatMul*'
_output_shapes
:��������� *
T0
�
<gradients_1/v/dense_6/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_6/MatMul_grad/MatMul_13^gradients_1/v/dense_6/MatMul_grad/tuple/group_deps*
_output_shapes

: *=
_class3
1/loc:@gradients_1/v/dense_6/MatMul_grad/MatMul_1*
T0
�
(gradients_1/v/dense_5/Relu_grad/ReluGradReluGrad:gradients_1/v/dense_6/MatMul_grad/tuple/control_dependencyv/dense_5/Relu*'
_output_shapes
:��������� *
T0
�
.gradients_1/v/dense_5/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_5/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
3gradients_1/v/dense_5/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_5/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_5/Relu_grad/ReluGrad
�
;gradients_1/v/dense_5/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_5/Relu_grad/ReluGrad4^gradients_1/v/dense_5/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense_5/Relu_grad/ReluGrad*'
_output_shapes
:��������� 
�
=gradients_1/v/dense_5/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_5/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_5/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/v/dense_5/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
�
(gradients_1/v/dense_5/MatMul_grad/MatMulMatMul;gradients_1/v/dense_5/BiasAdd_grad/tuple/control_dependencyv/dense_5/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
T0*
transpose_b(
�
*gradients_1/v/dense_5/MatMul_grad/MatMul_1MatMulv/dense_4/Relu;gradients_1/v/dense_5/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:@ *
T0*
transpose_b( 
�
2gradients_1/v/dense_5/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_5/MatMul_grad/MatMul+^gradients_1/v/dense_5/MatMul_grad/MatMul_1
�
:gradients_1/v/dense_5/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_5/MatMul_grad/MatMul3^gradients_1/v/dense_5/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense_5/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������@
�
<gradients_1/v/dense_5/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_5/MatMul_grad/MatMul_13^gradients_1/v/dense_5/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:@ *=
_class3
1/loc:@gradients_1/v/dense_5/MatMul_grad/MatMul_1
�
(gradients_1/v/dense_4/Relu_grad/ReluGradReluGrad:gradients_1/v/dense_5/MatMul_grad/tuple/control_dependencyv/dense_4/Relu*'
_output_shapes
:���������@*
T0
�
.gradients_1/v/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_4/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:@
�
3gradients_1/v/dense_4/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_4/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_4/Relu_grad/ReluGrad
�
;gradients_1/v/dense_4/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_4/Relu_grad/ReluGrad4^gradients_1/v/dense_4/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������@*;
_class1
/-loc:@gradients_1/v/dense_4/Relu_grad/ReluGrad*
T0
�
=gradients_1/v/dense_4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_4/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_4/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/v/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:@
�
(gradients_1/v/dense_4/MatMul_grad/MatMulMatMul;gradients_1/v/dense_4/BiasAdd_grad/tuple/control_dependencyv/dense_4/kernel/read*
T0*(
_output_shapes
:����������*
transpose_a( *
transpose_b(
�
*gradients_1/v/dense_4/MatMul_grad/MatMul_1MatMul	v/Squeeze;gradients_1/v/dense_4/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
_output_shapes
:	�@*
transpose_a(
�
2gradients_1/v/dense_4/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_4/MatMul_grad/MatMul+^gradients_1/v/dense_4/MatMul_grad/MatMul_1
�
:gradients_1/v/dense_4/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_4/MatMul_grad/MatMul3^gradients_1/v/dense_4/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense_4/MatMul_grad/MatMul*
T0*(
_output_shapes
:����������
�
<gradients_1/v/dense_4/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_4/MatMul_grad/MatMul_13^gradients_1/v/dense_4/MatMul_grad/tuple/group_deps*
_output_shapes
:	�@*=
_class3
1/loc:@gradients_1/v/dense_4/MatMul_grad/MatMul_1*
T0
q
 gradients_1/v/Squeeze_grad/ShapeShapev/dense_3/BiasAdd*
_output_shapes
:*
out_type0*
T0
�
"gradients_1/v/Squeeze_grad/ReshapeReshape:gradients_1/v/dense_4/MatMul_grad/tuple/control_dependency gradients_1/v/Squeeze_grad/Shape*
T0*
Tshape0*,
_output_shapes
:����������
�
.gradients_1/v/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients_1/v/Squeeze_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
�
3gradients_1/v/dense_3/BiasAdd_grad/tuple/group_depsNoOp#^gradients_1/v/Squeeze_grad/Reshape/^gradients_1/v/dense_3/BiasAdd_grad/BiasAddGrad
�
;gradients_1/v/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity"gradients_1/v/Squeeze_grad/Reshape4^gradients_1/v/dense_3/BiasAdd_grad/tuple/group_deps*,
_output_shapes
:����������*
T0*5
_class+
)'loc:@gradients_1/v/Squeeze_grad/Reshape
�
=gradients_1/v/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_3/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_3/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*A
_class7
53loc:@gradients_1/v/dense_3/BiasAdd_grad/BiasAddGrad*
T0
�
*gradients_1/v/dense_3/Tensordot_grad/ShapeShapev/dense_3/Tensordot/MatMul*
out_type0*
_output_shapes
:*
T0
�
,gradients_1/v/dense_3/Tensordot_grad/ReshapeReshape;gradients_1/v/dense_3/BiasAdd_grad/tuple/control_dependency*gradients_1/v/dense_3/Tensordot_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
2gradients_1/v/dense_3/Tensordot/MatMul_grad/MatMulMatMul,gradients_1/v/dense_3/Tensordot_grad/Reshapev/dense_3/Tensordot/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b(
�
4gradients_1/v/dense_3/Tensordot/MatMul_grad/MatMul_1MatMulv/dense_3/Tensordot/Reshape,gradients_1/v/dense_3/Tensordot_grad/Reshape*
transpose_b( *
T0*
transpose_a(*'
_output_shapes
:���������
�
<gradients_1/v/dense_3/Tensordot/MatMul_grad/tuple/group_depsNoOp3^gradients_1/v/dense_3/Tensordot/MatMul_grad/MatMul5^gradients_1/v/dense_3/Tensordot/MatMul_grad/MatMul_1
�
Dgradients_1/v/dense_3/Tensordot/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/v/dense_3/Tensordot/MatMul_grad/MatMul=^gradients_1/v/dense_3/Tensordot/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*E
_class;
97loc:@gradients_1/v/dense_3/Tensordot/MatMul_grad/MatMul
�
Fgradients_1/v/dense_3/Tensordot/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/v/dense_3/Tensordot/MatMul_grad/MatMul_1=^gradients_1/v/dense_3/Tensordot/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/v/dense_3/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:
�
2gradients_1/v/dense_3/Tensordot/Reshape_grad/ShapeShapev/dense_3/Tensordot/transpose*
T0*
out_type0*
_output_shapes
:
�
4gradients_1/v/dense_3/Tensordot/Reshape_grad/ReshapeReshapeDgradients_1/v/dense_3/Tensordot/MatMul_grad/tuple/control_dependency2gradients_1/v/dense_3/Tensordot/Reshape_grad/Shape*
T0*,
_output_shapes
:����������*
Tshape0
�
4gradients_1/v/dense_3/Tensordot/Reshape_1_grad/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
�
6gradients_1/v/dense_3/Tensordot/Reshape_1_grad/ReshapeReshapeFgradients_1/v/dense_3/Tensordot/MatMul_grad/tuple/control_dependency_14gradients_1/v/dense_3/Tensordot/Reshape_1_grad/Shape*
Tshape0*
T0*
_output_shapes

:
�
@gradients_1/v/dense_3/Tensordot/transpose_grad/InvertPermutationInvertPermutationv/dense_3/Tensordot/concat*
_output_shapes
:*
T0
�
8gradients_1/v/dense_3/Tensordot/transpose_grad/transpose	Transpose4gradients_1/v/dense_3/Tensordot/Reshape_grad/Reshape@gradients_1/v/dense_3/Tensordot/transpose_grad/InvertPermutation*,
_output_shapes
:����������*
T0*
Tperm0
�
Bgradients_1/v/dense_3/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation$v/dense_3/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
:gradients_1/v/dense_3/Tensordot/transpose_1_grad/transpose	Transpose6gradients_1/v/dense_3/Tensordot/Reshape_1_grad/ReshapeBgradients_1/v/dense_3/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
_output_shapes

:*
T0
�
(gradients_1/v/dense_2/Relu_grad/ReluGradReluGrad8gradients_1/v/dense_3/Tensordot/transpose_grad/transposev/dense_2/Relu*
T0*,
_output_shapes
:����������
�
.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_2/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
�
3gradients_1/v/dense_2/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_2/Relu_grad/ReluGrad
�
;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_2/Relu_grad/ReluGrad4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense_2/Relu_grad/ReluGrad*,
_output_shapes
:����������*
T0
�
=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
*gradients_1/v/dense_2/Tensordot_grad/ShapeShapev/dense_2/Tensordot/MatMul*
T0*
_output_shapes
:*
out_type0
�
,gradients_1/v/dense_2/Tensordot_grad/ReshapeReshape;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency*gradients_1/v/dense_2/Tensordot_grad/Shape*
T0*'
_output_shapes
:���������*
Tshape0
�
2gradients_1/v/dense_2/Tensordot/MatMul_grad/MatMulMatMul,gradients_1/v/dense_2/Tensordot_grad/Reshapev/dense_2/Tensordot/Reshape_1*
transpose_a( *
T0*'
_output_shapes
:���������*
transpose_b(
�
4gradients_1/v/dense_2/Tensordot/MatMul_grad/MatMul_1MatMulv/dense_2/Tensordot/Reshape,gradients_1/v/dense_2/Tensordot_grad/Reshape*
transpose_a(*
transpose_b( *'
_output_shapes
:���������*
T0
�
<gradients_1/v/dense_2/Tensordot/MatMul_grad/tuple/group_depsNoOp3^gradients_1/v/dense_2/Tensordot/MatMul_grad/MatMul5^gradients_1/v/dense_2/Tensordot/MatMul_grad/MatMul_1
�
Dgradients_1/v/dense_2/Tensordot/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/v/dense_2/Tensordot/MatMul_grad/MatMul=^gradients_1/v/dense_2/Tensordot/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*E
_class;
97loc:@gradients_1/v/dense_2/Tensordot/MatMul_grad/MatMul*
T0
�
Fgradients_1/v/dense_2/Tensordot/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/v/dense_2/Tensordot/MatMul_grad/MatMul_1=^gradients_1/v/dense_2/Tensordot/MatMul_grad/tuple/group_deps*G
_class=
;9loc:@gradients_1/v/dense_2/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
�
2gradients_1/v/dense_2/Tensordot/Reshape_grad/ShapeShapev/dense_2/Tensordot/transpose*
out_type0*
T0*
_output_shapes
:
�
4gradients_1/v/dense_2/Tensordot/Reshape_grad/ReshapeReshapeDgradients_1/v/dense_2/Tensordot/MatMul_grad/tuple/control_dependency2gradients_1/v/dense_2/Tensordot/Reshape_grad/Shape*
T0*
Tshape0*,
_output_shapes
:����������
�
4gradients_1/v/dense_2/Tensordot/Reshape_1_grad/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
�
6gradients_1/v/dense_2/Tensordot/Reshape_1_grad/ReshapeReshapeFgradients_1/v/dense_2/Tensordot/MatMul_grad/tuple/control_dependency_14gradients_1/v/dense_2/Tensordot/Reshape_1_grad/Shape*
Tshape0*
_output_shapes

:*
T0
�
@gradients_1/v/dense_2/Tensordot/transpose_grad/InvertPermutationInvertPermutationv/dense_2/Tensordot/concat*
T0*
_output_shapes
:
�
8gradients_1/v/dense_2/Tensordot/transpose_grad/transpose	Transpose4gradients_1/v/dense_2/Tensordot/Reshape_grad/Reshape@gradients_1/v/dense_2/Tensordot/transpose_grad/InvertPermutation*,
_output_shapes
:����������*
Tperm0*
T0
�
Bgradients_1/v/dense_2/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation$v/dense_2/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
:gradients_1/v/dense_2/Tensordot/transpose_1_grad/transpose	Transpose6gradients_1/v/dense_2/Tensordot/Reshape_1_grad/ReshapeBgradients_1/v/dense_2/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
T0*
_output_shapes

:
�
(gradients_1/v/dense_1/Relu_grad/ReluGradReluGrad8gradients_1/v/dense_2/Tensordot/transpose_grad/transposev/dense_1/Relu*
T0*,
_output_shapes
:����������
�
.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:
�
3gradients_1/v/dense_1/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_1/Relu_grad/ReluGrad
�
;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/Relu_grad/ReluGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense_1/Relu_grad/ReluGrad*,
_output_shapes
:����������
�
=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
*gradients_1/v/dense_1/Tensordot_grad/ShapeShapev/dense_1/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
�
,gradients_1/v/dense_1/Tensordot_grad/ReshapeReshape;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency*gradients_1/v/dense_1/Tensordot_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
2gradients_1/v/dense_1/Tensordot/MatMul_grad/MatMulMatMul,gradients_1/v/dense_1/Tensordot_grad/Reshapev/dense_1/Tensordot/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:��������� *
transpose_b(
�
4gradients_1/v/dense_1/Tensordot/MatMul_grad/MatMul_1MatMulv/dense_1/Tensordot/Reshape,gradients_1/v/dense_1/Tensordot_grad/Reshape*'
_output_shapes
:���������*
T0*
transpose_b( *
transpose_a(
�
<gradients_1/v/dense_1/Tensordot/MatMul_grad/tuple/group_depsNoOp3^gradients_1/v/dense_1/Tensordot/MatMul_grad/MatMul5^gradients_1/v/dense_1/Tensordot/MatMul_grad/MatMul_1
�
Dgradients_1/v/dense_1/Tensordot/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/v/dense_1/Tensordot/MatMul_grad/MatMul=^gradients_1/v/dense_1/Tensordot/MatMul_grad/tuple/group_deps*'
_output_shapes
:��������� *
T0*E
_class;
97loc:@gradients_1/v/dense_1/Tensordot/MatMul_grad/MatMul
�
Fgradients_1/v/dense_1/Tensordot/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/v/dense_1/Tensordot/MatMul_grad/MatMul_1=^gradients_1/v/dense_1/Tensordot/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

: *G
_class=
;9loc:@gradients_1/v/dense_1/Tensordot/MatMul_grad/MatMul_1
�
2gradients_1/v/dense_1/Tensordot/Reshape_grad/ShapeShapev/dense_1/Tensordot/transpose*
out_type0*
T0*
_output_shapes
:
�
4gradients_1/v/dense_1/Tensordot/Reshape_grad/ReshapeReshapeDgradients_1/v/dense_1/Tensordot/MatMul_grad/tuple/control_dependency2gradients_1/v/dense_1/Tensordot/Reshape_grad/Shape*
Tshape0*
T0*,
_output_shapes
:���������� 
�
4gradients_1/v/dense_1/Tensordot/Reshape_1_grad/ShapeConst*
dtype0*
valueB"       *
_output_shapes
:
�
6gradients_1/v/dense_1/Tensordot/Reshape_1_grad/ReshapeReshapeFgradients_1/v/dense_1/Tensordot/MatMul_grad/tuple/control_dependency_14gradients_1/v/dense_1/Tensordot/Reshape_1_grad/Shape*
Tshape0*
_output_shapes

: *
T0
�
@gradients_1/v/dense_1/Tensordot/transpose_grad/InvertPermutationInvertPermutationv/dense_1/Tensordot/concat*
_output_shapes
:*
T0
�
8gradients_1/v/dense_1/Tensordot/transpose_grad/transpose	Transpose4gradients_1/v/dense_1/Tensordot/Reshape_grad/Reshape@gradients_1/v/dense_1/Tensordot/transpose_grad/InvertPermutation*,
_output_shapes
:���������� *
Tperm0*
T0
�
Bgradients_1/v/dense_1/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation$v/dense_1/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
�
:gradients_1/v/dense_1/Tensordot/transpose_1_grad/transpose	Transpose6gradients_1/v/dense_1/Tensordot/Reshape_1_grad/ReshapeBgradients_1/v/dense_1/Tensordot/transpose_1_grad/InvertPermutation*
T0*
_output_shapes

: *
Tperm0
�
&gradients_1/v/dense/Relu_grad/ReluGradReluGrad8gradients_1/v/dense_1/Tensordot/transpose_grad/transposev/dense/Relu*,
_output_shapes
:���������� *
T0
�
,gradients_1/v/dense/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_1/v/dense/Relu_grad/ReluGrad*
_output_shapes
: *
data_formatNHWC*
T0
�
1gradients_1/v/dense/BiasAdd_grad/tuple/group_depsNoOp-^gradients_1/v/dense/BiasAdd_grad/BiasAddGrad'^gradients_1/v/dense/Relu_grad/ReluGrad
�
9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/Relu_grad/ReluGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*,
_output_shapes
:���������� *
T0*9
_class/
-+loc:@gradients_1/v/dense/Relu_grad/ReluGrad
�
;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Identity,gradients_1/v/dense/BiasAdd_grad/BiasAddGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
: *?
_class5
31loc:@gradients_1/v/dense/BiasAdd_grad/BiasAddGrad
�
(gradients_1/v/dense/Tensordot_grad/ShapeShapev/dense/Tensordot/MatMul*
T0*
out_type0*
_output_shapes
:
�
*gradients_1/v/dense/Tensordot_grad/ReshapeReshape9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency(gradients_1/v/dense/Tensordot_grad/Shape*
Tshape0*
T0*'
_output_shapes
:��������� 
�
0gradients_1/v/dense/Tensordot/MatMul_grad/MatMulMatMul*gradients_1/v/dense/Tensordot_grad/Reshapev/dense/Tensordot/Reshape_1*
transpose_b(*'
_output_shapes
:���������*
T0*
transpose_a( 
�
2gradients_1/v/dense/Tensordot/MatMul_grad/MatMul_1MatMulv/dense/Tensordot/Reshape*gradients_1/v/dense/Tensordot_grad/Reshape*
transpose_b( *
transpose_a(*
T0*'
_output_shapes
:��������� 
�
:gradients_1/v/dense/Tensordot/MatMul_grad/tuple/group_depsNoOp1^gradients_1/v/dense/Tensordot/MatMul_grad/MatMul3^gradients_1/v/dense/Tensordot/MatMul_grad/MatMul_1
�
Bgradients_1/v/dense/Tensordot/MatMul_grad/tuple/control_dependencyIdentity0gradients_1/v/dense/Tensordot/MatMul_grad/MatMul;^gradients_1/v/dense/Tensordot/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*C
_class9
75loc:@gradients_1/v/dense/Tensordot/MatMul_grad/MatMul*
T0
�
Dgradients_1/v/dense/Tensordot/MatMul_grad/tuple/control_dependency_1Identity2gradients_1/v/dense/Tensordot/MatMul_grad/MatMul_1;^gradients_1/v/dense/Tensordot/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

: *E
_class;
97loc:@gradients_1/v/dense/Tensordot/MatMul_grad/MatMul_1
�
2gradients_1/v/dense/Tensordot/Reshape_1_grad/ShapeConst*
valueB"       *
_output_shapes
:*
dtype0
�
4gradients_1/v/dense/Tensordot/Reshape_1_grad/ReshapeReshapeDgradients_1/v/dense/Tensordot/MatMul_grad/tuple/control_dependency_12gradients_1/v/dense/Tensordot/Reshape_1_grad/Shape*
Tshape0*
T0*
_output_shapes

: 
�
@gradients_1/v/dense/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation"v/dense/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
8gradients_1/v/dense/Tensordot/transpose_1_grad/transpose	Transpose4gradients_1/v/dense/Tensordot/Reshape_1_grad/Reshape@gradients_1/v/dense/Tensordot/transpose_1_grad/InvertPermutation*
_output_shapes

: *
Tperm0*
T0
�
beta1_power_1/initial_valueConst*
_output_shapes
: *
_class
loc:@v/dense/bias*
dtype0*
valueB
 *fff?
�
beta1_power_1
VariableV2*
	container *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@v/dense/bias*
shape: 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias*
validate_shape(
o
beta1_power_1/readIdentitybeta1_power_1*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
�
beta2_power_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*
_class
loc:@v/dense/bias
�
beta2_power_1
VariableV2*
_class
loc:@v/dense/bias*
shared_name *
_output_shapes
: *
dtype0*
shape: *
	container 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
_class
loc:@v/dense/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
o
beta2_power_1/readIdentitybeta2_power_1*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
�
%v/dense/kernel/Adam/Initializer/zerosConst*
dtype0*!
_class
loc:@v/dense/kernel*
_output_shapes

: *
valueB *    
�
v/dense/kernel/Adam
VariableV2*
_output_shapes

: *
	container *!
_class
loc:@v/dense/kernel*
dtype0*
shape
: *
shared_name 
�
v/dense/kernel/Adam/AssignAssignv/dense/kernel/Adam%v/dense/kernel/Adam/Initializer/zeros*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes

: 
�
v/dense/kernel/Adam/readIdentityv/dense/kernel/Adam*
T0*
_output_shapes

: *!
_class
loc:@v/dense/kernel
�
'v/dense/kernel/Adam_1/Initializer/zerosConst*
valueB *    *!
_class
loc:@v/dense/kernel*
_output_shapes

: *
dtype0
�
v/dense/kernel/Adam_1
VariableV2*
	container *!
_class
loc:@v/dense/kernel*
shape
: *
dtype0*
shared_name *
_output_shapes

: 
�
v/dense/kernel/Adam_1/AssignAssignv/dense/kernel/Adam_1'v/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

: *
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0
�
v/dense/kernel/Adam_1/readIdentityv/dense/kernel/Adam_1*
_output_shapes

: *
T0*!
_class
loc:@v/dense/kernel
�
#v/dense/bias/Adam/Initializer/zerosConst*
valueB *    *
_class
loc:@v/dense/bias*
dtype0*
_output_shapes
: 
�
v/dense/bias/Adam
VariableV2*
	container *
_output_shapes
: *
shape: *
_class
loc:@v/dense/bias*
shared_name *
dtype0
�
v/dense/bias/Adam/AssignAssignv/dense/bias/Adam#v/dense/bias/Adam/Initializer/zeros*
T0*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: 
{
v/dense/bias/Adam/readIdentityv/dense/bias/Adam*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
�
%v/dense/bias/Adam_1/Initializer/zerosConst*
valueB *    *
_class
loc:@v/dense/bias*
_output_shapes
: *
dtype0
�
v/dense/bias/Adam_1
VariableV2*
	container *
_class
loc:@v/dense/bias*
_output_shapes
: *
shared_name *
shape: *
dtype0
�
v/dense/bias/Adam_1/AssignAssignv/dense/bias/Adam_1%v/dense/bias/Adam_1/Initializer/zeros*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(

v/dense/bias/Adam_1/readIdentityv/dense/bias/Adam_1*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
'v/dense_1/kernel/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@v/dense_1/kernel*
valueB *    *
_output_shapes

: 
�
v/dense_1/kernel/Adam
VariableV2*
shape
: *
shared_name *
dtype0*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
	container 
�
v/dense_1/kernel/Adam/AssignAssignv/dense_1/kernel/Adam'v/dense_1/kernel/Adam/Initializer/zeros*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

: *
T0
�
v/dense_1/kernel/Adam/readIdentityv/dense_1/kernel/Adam*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
T0
�
)v/dense_1/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

: *
valueB *    *
dtype0*#
_class
loc:@v/dense_1/kernel
�
v/dense_1/kernel/Adam_1
VariableV2*
dtype0*#
_class
loc:@v/dense_1/kernel*
	container *
shape
: *
_output_shapes

: *
shared_name 
�
v/dense_1/kernel/Adam_1/AssignAssignv/dense_1/kernel/Adam_1)v/dense_1/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
_output_shapes

: *
validate_shape(*#
_class
loc:@v/dense_1/kernel
�
v/dense_1/kernel/Adam_1/readIdentityv/dense_1/kernel/Adam_1*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0
�
%v/dense_1/bias/Adam/Initializer/zerosConst*
valueB*    *!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
dtype0
�
v/dense_1/bias/Adam
VariableV2*
shared_name *
shape:*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
	container *
dtype0
�
v/dense_1/bias/Adam/AssignAssignv/dense_1/bias/Adam%v/dense_1/bias/Adam/Initializer/zeros*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:
�
v/dense_1/bias/Adam/readIdentityv/dense_1/bias/Adam*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:
�
'v/dense_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
dtype0*
valueB*    
�
v/dense_1/bias/Adam_1
VariableV2*
shared_name *!
_class
loc:@v/dense_1/bias*
dtype0*
_output_shapes
:*
shape:*
	container 
�
v/dense_1/bias/Adam_1/AssignAssignv/dense_1/bias/Adam_1'v/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
validate_shape(*
T0
�
v/dense_1/bias/Adam_1/readIdentityv/dense_1/bias/Adam_1*
T0*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
'v/dense_2/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*
dtype0*#
_class
loc:@v/dense_2/kernel*
valueB*    
�
v/dense_2/kernel/Adam
VariableV2*#
_class
loc:@v/dense_2/kernel*
dtype0*
	container *
shape
:*
shared_name *
_output_shapes

:
�
v/dense_2/kernel/Adam/AssignAssignv/dense_2/kernel/Adam'v/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
T0*
validate_shape(
�
v/dense_2/kernel/Adam/readIdentityv/dense_2/kernel/Adam*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
T0
�
)v/dense_2/kernel/Adam_1/Initializer/zerosConst*#
_class
loc:@v/dense_2/kernel*
dtype0*
valueB*    *
_output_shapes

:
�
v/dense_2/kernel/Adam_1
VariableV2*#
_class
loc:@v/dense_2/kernel*
shape
:*
	container *
_output_shapes

:*
shared_name *
dtype0
�
v/dense_2/kernel/Adam_1/AssignAssignv/dense_2/kernel/Adam_1)v/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel
�
v/dense_2/kernel/Adam_1/readIdentityv/dense_2/kernel/Adam_1*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
T0
�
%v/dense_2/bias/Adam/Initializer/zerosConst*!
_class
loc:@v/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
�
v/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
	container *!
_class
loc:@v/dense_2/bias*
shared_name *
dtype0*
shape:
�
v/dense_2/bias/Adam/AssignAssignv/dense_2/bias/Adam%v/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(
�
v/dense_2/bias/Adam/readIdentityv/dense_2/bias/Adam*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
'v/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
dtype0*!
_class
loc:@v/dense_2/bias
�
v/dense_2/bias/Adam_1
VariableV2*
_output_shapes
:*
shape:*
shared_name *!
_class
loc:@v/dense_2/bias*
dtype0*
	container 
�
v/dense_2/bias/Adam_1/AssignAssignv/dense_2/bias/Adam_1'v/dense_2/bias/Adam_1/Initializer/zeros*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
�
v/dense_2/bias/Adam_1/readIdentityv/dense_2/bias/Adam_1*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0
�
'v/dense_3/kernel/Adam/Initializer/zerosConst*
dtype0*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
valueB*    
�
v/dense_3/kernel/Adam
VariableV2*#
_class
loc:@v/dense_3/kernel*
	container *
_output_shapes

:*
shared_name *
shape
:*
dtype0
�
v/dense_3/kernel/Adam/AssignAssignv/dense_3/kernel/Adam'v/dense_3/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:
�
v/dense_3/kernel/Adam/readIdentityv/dense_3/kernel/Adam*#
_class
loc:@v/dense_3/kernel*
T0*
_output_shapes

:
�
)v/dense_3/kernel/Adam_1/Initializer/zerosConst*#
_class
loc:@v/dense_3/kernel*
dtype0*
valueB*    *
_output_shapes

:
�
v/dense_3/kernel/Adam_1
VariableV2*
_output_shapes

:*
shared_name *
	container *#
_class
loc:@v/dense_3/kernel*
dtype0*
shape
:
�
v/dense_3/kernel/Adam_1/AssignAssignv/dense_3/kernel/Adam_1)v/dense_3/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
T0*#
_class
loc:@v/dense_3/kernel*
use_locking(*
validate_shape(
�
v/dense_3/kernel/Adam_1/readIdentityv/dense_3/kernel/Adam_1*
_output_shapes

:*
T0*#
_class
loc:@v/dense_3/kernel
�
%v/dense_3/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes
:*!
_class
loc:@v/dense_3/bias
�
v/dense_3/bias/Adam
VariableV2*
shared_name *
dtype0*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
shape:*
	container 
�
v/dense_3/bias/Adam/AssignAssignv/dense_3/bias/Adam%v/dense_3/bias/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_3/bias*
_output_shapes
:
�
v/dense_3/bias/Adam/readIdentityv/dense_3/bias/Adam*
T0*!
_class
loc:@v/dense_3/bias*
_output_shapes
:
�
'v/dense_3/bias/Adam_1/Initializer/zerosConst*
valueB*    *!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
dtype0
�
v/dense_3/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
	container *
shared_name *!
_class
loc:@v/dense_3/bias*
shape:
�
v/dense_3/bias/Adam_1/AssignAssignv/dense_3/bias/Adam_1'v/dense_3/bias/Adam_1/Initializer/zeros*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
use_locking(*
validate_shape(*
T0
�
v/dense_3/bias/Adam_1/readIdentityv/dense_3/bias/Adam_1*!
_class
loc:@v/dense_3/bias*
T0*
_output_shapes
:
�
7v/dense_4/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB" 
  @   *
_output_shapes
:*
dtype0*#
_class
loc:@v/dense_4/kernel
�
-v/dense_4/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *#
_class
loc:@v/dense_4/kernel
�
'v/dense_4/kernel/Adam/Initializer/zerosFill7v/dense_4/kernel/Adam/Initializer/zeros/shape_as_tensor-v/dense_4/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	�@*

index_type0*#
_class
loc:@v/dense_4/kernel*
T0
�
v/dense_4/kernel/Adam
VariableV2*
shape:	�@*
dtype0*#
_class
loc:@v/dense_4/kernel*
	container *
_output_shapes
:	�@*
shared_name 
�
v/dense_4/kernel/Adam/AssignAssignv/dense_4/kernel/Adam'v/dense_4/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel
�
v/dense_4/kernel/Adam/readIdentityv/dense_4/kernel/Adam*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
T0
�
9v/dense_4/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*#
_class
loc:@v/dense_4/kernel*
valueB" 
  @   *
dtype0
�
/v/dense_4/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *#
_class
loc:@v/dense_4/kernel*
dtype0
�
)v/dense_4/kernel/Adam_1/Initializer/zerosFill9v/dense_4/kernel/Adam_1/Initializer/zeros/shape_as_tensor/v/dense_4/kernel/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*

index_type0
�
v/dense_4/kernel/Adam_1
VariableV2*#
_class
loc:@v/dense_4/kernel*
	container *
shared_name *
dtype0*
shape:	�@*
_output_shapes
:	�@
�
v/dense_4/kernel/Adam_1/AssignAssignv/dense_4/kernel/Adam_1)v/dense_4/kernel/Adam_1/Initializer/zeros*
use_locking(*#
_class
loc:@v/dense_4/kernel*
validate_shape(*
_output_shapes
:	�@*
T0
�
v/dense_4/kernel/Adam_1/readIdentityv/dense_4/kernel/Adam_1*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*
T0
�
%v/dense_4/bias/Adam/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@*
dtype0*!
_class
loc:@v/dense_4/bias
�
v/dense_4/bias/Adam
VariableV2*
shared_name *!
_class
loc:@v/dense_4/bias*
shape:@*
_output_shapes
:@*
	container *
dtype0
�
v/dense_4/bias/Adam/AssignAssignv/dense_4/bias/Adam%v/dense_4/bias/Adam/Initializer/zeros*
use_locking(*!
_class
loc:@v/dense_4/bias*
T0*
validate_shape(*
_output_shapes
:@
�
v/dense_4/bias/Adam/readIdentityv/dense_4/bias/Adam*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias
�
'v/dense_4/bias/Adam_1/Initializer/zerosConst*
valueB@*    *!
_class
loc:@v/dense_4/bias*
dtype0*
_output_shapes
:@
�
v/dense_4/bias/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
shared_name *
	container 
�
v/dense_4/bias/Adam_1/AssignAssignv/dense_4/bias/Adam_1'v/dense_4/bias/Adam_1/Initializer/zeros*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
T0*
use_locking(*
validate_shape(
�
v/dense_4/bias/Adam_1/readIdentityv/dense_4/bias/Adam_1*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
T0
�
7v/dense_5/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@       *#
_class
loc:@v/dense_5/kernel
�
-v/dense_5/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*#
_class
loc:@v/dense_5/kernel*
_output_shapes
: *
valueB
 *    
�
'v/dense_5/kernel/Adam/Initializer/zerosFill7v/dense_5/kernel/Adam/Initializer/zeros/shape_as_tensor-v/dense_5/kernel/Adam/Initializer/zeros/Const*
T0*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel*

index_type0
�
v/dense_5/kernel/Adam
VariableV2*
shape
:@ *
_output_shapes

:@ *
	container *#
_class
loc:@v/dense_5/kernel*
dtype0*
shared_name 
�
v/dense_5/kernel/Adam/AssignAssignv/dense_5/kernel/Adam'v/dense_5/kernel/Adam/Initializer/zeros*
T0*
_output_shapes

:@ *
use_locking(*#
_class
loc:@v/dense_5/kernel*
validate_shape(
�
v/dense_5/kernel/Adam/readIdentityv/dense_5/kernel/Adam*
T0*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel
�
9v/dense_5/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@       *#
_class
loc:@v/dense_5/kernel*
_output_shapes
:*
dtype0
�
/v/dense_5/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *#
_class
loc:@v/dense_5/kernel
�
)v/dense_5/kernel/Adam_1/Initializer/zerosFill9v/dense_5/kernel/Adam_1/Initializer/zeros/shape_as_tensor/v/dense_5/kernel/Adam_1/Initializer/zeros/Const*#
_class
loc:@v/dense_5/kernel*
T0*

index_type0*
_output_shapes

:@ 
�
v/dense_5/kernel/Adam_1
VariableV2*
	container *
shared_name *
shape
:@ *
dtype0*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel
�
v/dense_5/kernel/Adam_1/AssignAssignv/dense_5/kernel/Adam_1)v/dense_5/kernel/Adam_1/Initializer/zeros*
validate_shape(*#
_class
loc:@v/dense_5/kernel*
T0*
use_locking(*
_output_shapes

:@ 
�
v/dense_5/kernel/Adam_1/readIdentityv/dense_5/kernel/Adam_1*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel*
T0
�
%v/dense_5/bias/Adam/Initializer/zerosConst*
valueB *    *
dtype0*
_output_shapes
: *!
_class
loc:@v/dense_5/bias
�
v/dense_5/bias/Adam
VariableV2*
shared_name *
_output_shapes
: *!
_class
loc:@v/dense_5/bias*
dtype0*
	container *
shape: 
�
v/dense_5/bias/Adam/AssignAssignv/dense_5/bias/Adam%v/dense_5/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
: *!
_class
loc:@v/dense_5/bias*
T0*
use_locking(
�
v/dense_5/bias/Adam/readIdentityv/dense_5/bias/Adam*!
_class
loc:@v/dense_5/bias*
T0*
_output_shapes
: 
�
'v/dense_5/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *!
_class
loc:@v/dense_5/bias
�
v/dense_5/bias/Adam_1
VariableV2*
dtype0*!
_class
loc:@v/dense_5/bias*
	container *
shape: *
shared_name *
_output_shapes
: 
�
v/dense_5/bias/Adam_1/AssignAssignv/dense_5/bias/Adam_1'v/dense_5/bias/Adam_1/Initializer/zeros*
T0*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
use_locking(*
validate_shape(
�
v/dense_5/bias/Adam_1/readIdentityv/dense_5/bias/Adam_1*
T0*
_output_shapes
: *!
_class
loc:@v/dense_5/bias
�
'v/dense_6/kernel/Adam/Initializer/zerosConst*
valueB *    *
dtype0*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: 
�
v/dense_6/kernel/Adam
VariableV2*
	container *
dtype0*
_output_shapes

: *
shape
: *#
_class
loc:@v/dense_6/kernel*
shared_name 
�
v/dense_6/kernel/Adam/AssignAssignv/dense_6/kernel/Adam'v/dense_6/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
T0*
use_locking(
�
v/dense_6/kernel/Adam/readIdentityv/dense_6/kernel/Adam*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
T0
�
)v/dense_6/kernel/Adam_1/Initializer/zerosConst*
valueB *    *
dtype0*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: 
�
v/dense_6/kernel/Adam_1
VariableV2*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
shape
: *
dtype0*
	container *
shared_name 
�
v/dense_6/kernel/Adam_1/AssignAssignv/dense_6/kernel/Adam_1)v/dense_6/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel
�
v/dense_6/kernel/Adam_1/readIdentityv/dense_6/kernel/Adam_1*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
T0
�
%v/dense_6/bias/Adam/Initializer/zerosConst*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
valueB*    *
dtype0
�
v/dense_6/bias/Adam
VariableV2*
	container *!
_class
loc:@v/dense_6/bias*
dtype0*
_output_shapes
:*
shape:*
shared_name 
�
v/dense_6/bias/Adam/AssignAssignv/dense_6/bias/Adam%v/dense_6/bias/Adam/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
use_locking(
�
v/dense_6/bias/Adam/readIdentityv/dense_6/bias/Adam*
T0*!
_class
loc:@v/dense_6/bias*
_output_shapes
:
�
'v/dense_6/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@v/dense_6/bias*
_output_shapes
:*
valueB*    *
dtype0
�
v/dense_6/bias/Adam_1
VariableV2*!
_class
loc:@v/dense_6/bias*
_output_shapes
:*
shape:*
dtype0*
	container *
shared_name 
�
v/dense_6/bias/Adam_1/AssignAssignv/dense_6/bias/Adam_1'v/dense_6/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_6/bias
�
v/dense_6/bias/Adam_1/readIdentityv/dense_6/bias/Adam_1*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
T0
�
'v/dense_7/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
valueB*    
�
v/dense_7/kernel/Adam
VariableV2*
shared_name *
	container *
shape
:*#
_class
loc:@v/dense_7/kernel*
dtype0*
_output_shapes

:
�
v/dense_7/kernel/Adam/AssignAssignv/dense_7/kernel/Adam'v/dense_7/kernel/Adam/Initializer/zeros*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
T0*
_output_shapes

:*
use_locking(
�
v/dense_7/kernel/Adam/readIdentityv/dense_7/kernel/Adam*
_output_shapes

:*
T0*#
_class
loc:@v/dense_7/kernel
�
)v/dense_7/kernel/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *#
_class
loc:@v/dense_7/kernel*
_output_shapes

:
�
v/dense_7/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
shape
:*
shared_name *
	container 
�
v/dense_7/kernel/Adam_1/AssignAssignv/dense_7/kernel/Adam_1)v/dense_7/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
T0*
use_locking(*#
_class
loc:@v/dense_7/kernel
�
v/dense_7/kernel/Adam_1/readIdentityv/dense_7/kernel/Adam_1*#
_class
loc:@v/dense_7/kernel*
_output_shapes

:*
T0
�
%v/dense_7/bias/Adam/Initializer/zerosConst*!
_class
loc:@v/dense_7/bias*
dtype0*
_output_shapes
:*
valueB*    
�
v/dense_7/bias/Adam
VariableV2*
shape:*
_output_shapes
:*
shared_name *
	container *!
_class
loc:@v/dense_7/bias*
dtype0
�
v/dense_7/bias/Adam/AssignAssignv/dense_7/bias/Adam%v/dense_7/bias/Adam/Initializer/zeros*!
_class
loc:@v/dense_7/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
�
v/dense_7/bias/Adam/readIdentityv/dense_7/bias/Adam*!
_class
loc:@v/dense_7/bias*
_output_shapes
:*
T0
�
'v/dense_7/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@v/dense_7/bias*
valueB*    
�
v/dense_7/bias/Adam_1
VariableV2*
_output_shapes
:*
shape:*!
_class
loc:@v/dense_7/bias*
dtype0*
shared_name *
	container 
�
v/dense_7/bias/Adam_1/AssignAssignv/dense_7/bias/Adam_1'v/dense_7/bias/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_7/bias
�
v/dense_7/bias/Adam_1/readIdentityv/dense_7/bias/Adam_1*
_output_shapes
:*
T0*!
_class
loc:@v/dense_7/bias
Y
Adam_1/learning_rateConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
Q
Adam_1/beta1Const*
_output_shapes
: *
dtype0*
valueB
 *fff?
Q
Adam_1/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
&Adam_1/update_v/dense/kernel/ApplyAdam	ApplyAdamv/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon8gradients_1/v/dense/Tensordot/transpose_1_grad/transpose*!
_class
loc:@v/dense/kernel*
use_locking( *
_output_shapes

: *
use_nesterov( *
T0
�
$Adam_1/update_v/dense/bias/ApplyAdam	ApplyAdamv/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1*
_class
loc:@v/dense/bias*
use_nesterov( *
T0*
use_locking( *
_output_shapes
: 
�
(Adam_1/update_v/dense_1/kernel/ApplyAdam	ApplyAdamv/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon:gradients_1/v/dense_1/Tensordot/transpose_1_grad/transpose*
_output_shapes

: *
use_nesterov( *#
_class
loc:@v/dense_1/kernel*
T0*
use_locking( 
�
&Adam_1/update_v/dense_1/bias/ApplyAdam	ApplyAdamv/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
use_locking( 
�
(Adam_1/update_v/dense_2/kernel/ApplyAdam	ApplyAdamv/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon:gradients_1/v/dense_2/Tensordot/transpose_1_grad/transpose*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
use_locking( *
T0*
use_nesterov( 
�
&Adam_1/update_v/dense_2/bias/ApplyAdam	ApplyAdamv/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *!
_class
loc:@v/dense_2/bias*
use_nesterov( *
_output_shapes
:*
T0
�
(Adam_1/update_v/dense_3/kernel/ApplyAdam	ApplyAdamv/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon:gradients_1/v/dense_3/Tensordot/transpose_1_grad/transpose*
use_nesterov( *
T0*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
use_locking( 
�
&Adam_1/update_v/dense_3/bias/ApplyAdam	ApplyAdamv/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_3/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
use_nesterov( *
T0*!
_class
loc:@v/dense_3/bias*
use_locking( 
�
(Adam_1/update_v/dense_4/kernel/ApplyAdam	ApplyAdamv/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_1/v/dense_4/MatMul_grad/tuple/control_dependency_1*#
_class
loc:@v/dense_4/kernel*
T0*
_output_shapes
:	�@*
use_nesterov( *
use_locking( 
�
&Adam_1/update_v/dense_4/bias/ApplyAdam	ApplyAdamv/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_4/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:@*
use_locking( *
T0*!
_class
loc:@v/dense_4/bias*
use_nesterov( 
�
(Adam_1/update_v/dense_5/kernel/ApplyAdam	ApplyAdamv/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_1/v/dense_5/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:@ *
use_nesterov( *
use_locking( *
T0*#
_class
loc:@v/dense_5/kernel
�
&Adam_1/update_v/dense_5/bias/ApplyAdam	ApplyAdamv/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_5/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes
: *!
_class
loc:@v/dense_5/bias*
T0
�
(Adam_1/update_v/dense_6/kernel/ApplyAdam	ApplyAdamv/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_1/v/dense_6/MatMul_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
T0*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: 
�
&Adam_1/update_v/dense_6/bias/ApplyAdam	ApplyAdamv/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_6/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
T0*!
_class
loc:@v/dense_6/bias*
use_locking( 
�
(Adam_1/update_v/dense_7/kernel/ApplyAdam	ApplyAdamv/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_1/v/dense_7/MatMul_grad/tuple/control_dependency_1*
T0*
use_nesterov( *#
_class
loc:@v/dense_7/kernel*
use_locking( *
_output_shapes

:
�
&Adam_1/update_v/dense_7/bias/ApplyAdam	ApplyAdamv/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_7/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
use_nesterov( *!
_class
loc:@v/dense_7/bias*
_output_shapes
:
�

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam'^Adam_1/update_v/dense_3/bias/ApplyAdam)^Adam_1/update_v/dense_3/kernel/ApplyAdam'^Adam_1/update_v/dense_4/bias/ApplyAdam)^Adam_1/update_v/dense_4/kernel/ApplyAdam'^Adam_1/update_v/dense_5/bias/ApplyAdam)^Adam_1/update_v/dense_5/kernel/ApplyAdam'^Adam_1/update_v/dense_6/bias/ApplyAdam)^Adam_1/update_v/dense_6/kernel/ApplyAdam'^Adam_1/update_v/dense_7/bias/ApplyAdam)^Adam_1/update_v/dense_7/kernel/ApplyAdam*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*
_output_shapes
: *
use_locking( *
_class
loc:@v/dense/bias*
validate_shape(
�
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam'^Adam_1/update_v/dense_3/bias/ApplyAdam)^Adam_1/update_v/dense_3/kernel/ApplyAdam'^Adam_1/update_v/dense_4/bias/ApplyAdam)^Adam_1/update_v/dense_4/kernel/ApplyAdam'^Adam_1/update_v/dense_5/bias/ApplyAdam)^Adam_1/update_v/dense_5/kernel/ApplyAdam'^Adam_1/update_v/dense_6/bias/ApplyAdam)^Adam_1/update_v/dense_6/kernel/ApplyAdam'^Adam_1/update_v/dense_7/bias/ApplyAdam)^Adam_1/update_v/dense_7/kernel/ApplyAdam*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam'^Adam_1/update_v/dense_3/bias/ApplyAdam)^Adam_1/update_v/dense_3/kernel/ApplyAdam'^Adam_1/update_v/dense_4/bias/ApplyAdam)^Adam_1/update_v/dense_4/kernel/ApplyAdam'^Adam_1/update_v/dense_5/bias/ApplyAdam)^Adam_1/update_v/dense_5/kernel/ApplyAdam'^Adam_1/update_v/dense_6/bias/ApplyAdam)^Adam_1/update_v/dense_6/kernel/ApplyAdam'^Adam_1/update_v/dense_7/bias/ApplyAdam)^Adam_1/update_v/dense_7/kernel/ApplyAdam
�
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^pi/dense_3/bias/Adam/Assign^pi/dense_3/bias/Adam_1/Assign^pi/dense_3/bias/Assign^pi/dense_3/kernel/Adam/Assign ^pi/dense_3/kernel/Adam_1/Assign^pi/dense_3/kernel/Assign^v/dense/bias/Adam/Assign^v/dense/bias/Adam_1/Assign^v/dense/bias/Assign^v/dense/kernel/Adam/Assign^v/dense/kernel/Adam_1/Assign^v/dense/kernel/Assign^v/dense_1/bias/Adam/Assign^v/dense_1/bias/Adam_1/Assign^v/dense_1/bias/Assign^v/dense_1/kernel/Adam/Assign^v/dense_1/kernel/Adam_1/Assign^v/dense_1/kernel/Assign^v/dense_2/bias/Adam/Assign^v/dense_2/bias/Adam_1/Assign^v/dense_2/bias/Assign^v/dense_2/kernel/Adam/Assign^v/dense_2/kernel/Adam_1/Assign^v/dense_2/kernel/Assign^v/dense_3/bias/Adam/Assign^v/dense_3/bias/Adam_1/Assign^v/dense_3/bias/Assign^v/dense_3/kernel/Adam/Assign^v/dense_3/kernel/Adam_1/Assign^v/dense_3/kernel/Assign^v/dense_4/bias/Adam/Assign^v/dense_4/bias/Adam_1/Assign^v/dense_4/bias/Assign^v/dense_4/kernel/Adam/Assign^v/dense_4/kernel/Adam_1/Assign^v/dense_4/kernel/Assign^v/dense_5/bias/Adam/Assign^v/dense_5/bias/Adam_1/Assign^v/dense_5/bias/Assign^v/dense_5/kernel/Adam/Assign^v/dense_5/kernel/Adam_1/Assign^v/dense_5/kernel/Assign^v/dense_6/bias/Adam/Assign^v/dense_6/bias/Adam_1/Assign^v/dense_6/bias/Assign^v/dense_6/kernel/Adam/Assign^v/dense_6/kernel/Adam_1/Assign^v/dense_6/kernel/Assign^v/dense_7/bias/Adam/Assign^v/dense_7/bias/Adam_1/Assign^v/dense_7/bias/Assign^v/dense_7/kernel/Adam/Assign^v/dense_7/kernel/Adam_1/Assign^v/dense_7/kernel/Assign
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_def6c9e7683e465480a859211c7e3da6/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
N*
_output_shapes
:*
T0*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
save/RestoreV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L
�
save/AssignAssignbeta1_powersave/RestoreV2* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
�
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
_output_shapes
: *
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias
�
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
: 
�
save/Assign_4Assignpi/dense/biassave/RestoreV2:4*
T0*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save/Assign_5Assignpi/dense/bias/Adamsave/RestoreV2:5*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
�
save/Assign_6Assignpi/dense/bias/Adam_1save/RestoreV2:6*
_output_shapes
: *
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save/Assign_7Assignpi/dense/kernelsave/RestoreV2:7*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
T0*
validate_shape(
�
save/Assign_8Assignpi/dense/kernel/Adamsave/RestoreV2:8*
_output_shapes

: *
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
�
save/Assign_9Assignpi/dense/kernel/Adam_1save/RestoreV2:9*
_output_shapes

: *
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel
�
save/Assign_10Assignpi/dense_1/biassave/RestoreV2:10*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:
�
save/Assign_11Assignpi/dense_1/bias/Adamsave/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save/Assign_12Assignpi/dense_1/bias/Adam_1save/RestoreV2:12*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes
:
�
save/Assign_13Assignpi/dense_1/kernelsave/RestoreV2:13*
use_locking(*
_output_shapes

: *
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0
�
save/Assign_14Assignpi/dense_1/kernel/Adamsave/RestoreV2:14*
use_locking(*
validate_shape(*
T0*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel
�
save/Assign_15Assignpi/dense_1/kernel/Adam_1save/RestoreV2:15*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

: *
use_locking(
�
save/Assign_16Assignpi/dense_2/biassave/RestoreV2:16*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(
�
save/Assign_17Assignpi/dense_2/bias/Adamsave/RestoreV2:17*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(
�
save/Assign_18Assignpi/dense_2/bias/Adam_1save/RestoreV2:18*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
�
save/Assign_19Assignpi/dense_2/kernelsave/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:*
T0
�
save/Assign_20Assignpi/dense_2/kernel/Adamsave/RestoreV2:20*
validate_shape(*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
T0
�
save/Assign_21Assignpi/dense_2/kernel/Adam_1save/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:*
validate_shape(*
use_locking(
�
save/Assign_22Assignpi/dense_3/biassave/RestoreV2:22*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save/Assign_23Assignpi/dense_3/bias/Adamsave/RestoreV2:23*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
save/Assign_24Assignpi/dense_3/bias/Adam_1save/RestoreV2:24*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
save/Assign_25Assignpi/dense_3/kernelsave/RestoreV2:25*
_output_shapes

:*
validate_shape(*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
T0
�
save/Assign_26Assignpi/dense_3/kernel/Adamsave/RestoreV2:26*
_output_shapes

:*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
T0*
validate_shape(
�
save/Assign_27Assignpi/dense_3/kernel/Adam_1save/RestoreV2:27*$
_class
loc:@pi/dense_3/kernel*
T0*
_output_shapes

:*
use_locking(*
validate_shape(
�
save/Assign_28Assignv/dense/biassave/RestoreV2:28*
_class
loc:@v/dense/bias*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_29Assignv/dense/bias/Adamsave/RestoreV2:29*
validate_shape(*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias*
T0
�
save/Assign_30Assignv/dense/bias/Adam_1save/RestoreV2:30*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
�
save/Assign_31Assignv/dense/kernelsave/RestoreV2:31*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

: *
validate_shape(
�
save/Assign_32Assignv/dense/kernel/Adamsave/RestoreV2:32*
_output_shapes

: *!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
T0
�
save/Assign_33Assignv/dense/kernel/Adam_1save/RestoreV2:33*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*!
_class
loc:@v/dense/kernel
�
save/Assign_34Assignv/dense_1/biassave/RestoreV2:34*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_1/bias
�
save/Assign_35Assignv/dense_1/bias/Adamsave/RestoreV2:35*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
save/Assign_36Assignv/dense_1/bias/Adam_1save/RestoreV2:36*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
�
save/Assign_37Assignv/dense_1/kernelsave/RestoreV2:37*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(*
T0
�
save/Assign_38Assignv/dense_1/kernel/Adamsave/RestoreV2:38*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
T0
�
save/Assign_39Assignv/dense_1/kernel/Adam_1save/RestoreV2:39*
validate_shape(*
use_locking(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0
�
save/Assign_40Assignv/dense_2/biassave/RestoreV2:40*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_41Assignv/dense_2/bias/Adamsave/RestoreV2:41*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
�
save/Assign_42Assignv/dense_2/bias/Adam_1save/RestoreV2:42*
_output_shapes
:*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save/Assign_43Assignv/dense_2/kernelsave/RestoreV2:43*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:
�
save/Assign_44Assignv/dense_2/kernel/Adamsave/RestoreV2:44*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
validate_shape(
�
save/Assign_45Assignv/dense_2/kernel/Adam_1save/RestoreV2:45*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:
�
save/Assign_46Assignv/dense_3/biassave/RestoreV2:46*!
_class
loc:@v/dense_3/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
�
save/Assign_47Assignv/dense_3/bias/Adamsave/RestoreV2:47*
T0*
validate_shape(*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
use_locking(
�
save/Assign_48Assignv/dense_3/bias/Adam_1save/RestoreV2:48*!
_class
loc:@v/dense_3/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_49Assignv/dense_3/kernelsave/RestoreV2:49*
_output_shapes

:*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_3/kernel
�
save/Assign_50Assignv/dense_3/kernel/Adamsave/RestoreV2:50*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
use_locking(*
_output_shapes

:*
T0
�
save/Assign_51Assignv/dense_3/kernel/Adam_1save/RestoreV2:51*
validate_shape(*
use_locking(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel
�
save/Assign_52Assignv/dense_4/biassave/RestoreV2:52*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_4/bias
�
save/Assign_53Assignv/dense_4/bias/Adamsave/RestoreV2:53*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_4/bias*
T0*
_output_shapes
:@
�
save/Assign_54Assignv/dense_4/bias/Adam_1save/RestoreV2:54*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_4/bias
�
save/Assign_55Assignv/dense_4/kernelsave/RestoreV2:55*
T0*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
use_locking(*
validate_shape(
�
save/Assign_56Assignv/dense_4/kernel/Adamsave/RestoreV2:56*
use_locking(*
T0*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*
validate_shape(
�
save/Assign_57Assignv/dense_4/kernel/Adam_1save/RestoreV2:57*
_output_shapes
:	�@*
T0*
use_locking(*#
_class
loc:@v/dense_4/kernel*
validate_shape(
�
save/Assign_58Assignv/dense_5/biassave/RestoreV2:58*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_5/bias
�
save/Assign_59Assignv/dense_5/bias/Adamsave/RestoreV2:59*
validate_shape(*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
use_locking(*
T0
�
save/Assign_60Assignv/dense_5/bias/Adam_1save/RestoreV2:60*
use_locking(*
_output_shapes
: *
validate_shape(*!
_class
loc:@v/dense_5/bias*
T0
�
save/Assign_61Assignv/dense_5/kernelsave/RestoreV2:61*
T0*
use_locking(*#
_class
loc:@v/dense_5/kernel*
validate_shape(*
_output_shapes

:@ 
�
save/Assign_62Assignv/dense_5/kernel/Adamsave/RestoreV2:62*
_output_shapes

:@ *
use_locking(*#
_class
loc:@v/dense_5/kernel*
T0*
validate_shape(
�
save/Assign_63Assignv/dense_5/kernel/Adam_1save/RestoreV2:63*
_output_shapes

:@ *
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_5/kernel
�
save/Assign_64Assignv/dense_6/biassave/RestoreV2:64*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_6/bias*
use_locking(
�
save/Assign_65Assignv/dense_6/bias/Adamsave/RestoreV2:65*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_6/bias*
validate_shape(
�
save/Assign_66Assignv/dense_6/bias/Adam_1save/RestoreV2:66*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_6/bias
�
save/Assign_67Assignv/dense_6/kernelsave/RestoreV2:67*
T0*#
_class
loc:@v/dense_6/kernel*
validate_shape(*
_output_shapes

: *
use_locking(
�
save/Assign_68Assignv/dense_6/kernel/Adamsave/RestoreV2:68*
use_locking(*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
T0*
validate_shape(
�
save/Assign_69Assignv/dense_6/kernel/Adam_1save/RestoreV2:69*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: 
�
save/Assign_70Assignv/dense_7/biassave/RestoreV2:70*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_7/bias*
use_locking(*
T0
�
save/Assign_71Assignv/dense_7/bias/Adamsave/RestoreV2:71*!
_class
loc:@v/dense_7/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
�
save/Assign_72Assignv/dense_7/bias/Adam_1save/RestoreV2:72*
use_locking(*
T0*!
_class
loc:@v/dense_7/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_73Assignv/dense_7/kernelsave/RestoreV2:73*
validate_shape(*
_output_shapes

:*
T0*
use_locking(*#
_class
loc:@v/dense_7/kernel
�
save/Assign_74Assignv/dense_7/kernel/Adamsave/RestoreV2:74*
use_locking(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
validate_shape(
�
save/Assign_75Assignv/dense_7/kernel/Adam_1save/RestoreV2:75*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
T0*
use_locking(
�

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
shape: *
dtype0
�
save_1/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_054094d1bf724be2bf0c89096967807e/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_1/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: *
T0
�
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
_output_shapes
:*
T0*

axis *
N
�
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
�
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
_output_shapes
: *
T0
�
save_1/RestoreV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_1/AssignAssignbeta1_powersave_1/RestoreV2* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
T0
�
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
validate_shape(*
T0*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
save_1/Assign_4Assignpi/dense/biassave_1/RestoreV2:4*
validate_shape(*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(
�
save_1/Assign_5Assignpi/dense/bias/Adamsave_1/RestoreV2:5*
validate_shape(*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(
�
save_1/Assign_6Assignpi/dense/bias/Adam_1save_1/RestoreV2:6* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
�
save_1/Assign_7Assignpi/dense/kernelsave_1/RestoreV2:7*
_output_shapes

: *
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
�
save_1/Assign_8Assignpi/dense/kernel/Adamsave_1/RestoreV2:8*
_output_shapes

: *
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(
�
save_1/Assign_9Assignpi/dense/kernel/Adam_1save_1/RestoreV2:9*
T0*
use_locking(*
validate_shape(*
_output_shapes

: *"
_class
loc:@pi/dense/kernel
�
save_1/Assign_10Assignpi/dense_1/biassave_1/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_1/Assign_11Assignpi/dense_1/bias/Adamsave_1/RestoreV2:11*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
_output_shapes
:
�
save_1/Assign_12Assignpi/dense_1/bias/Adam_1save_1/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
�
save_1/Assign_13Assignpi/dense_1/kernelsave_1/RestoreV2:13*
validate_shape(*
use_locking(*
T0*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel
�
save_1/Assign_14Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

: *
validate_shape(*
use_locking(
�
save_1/Assign_15Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:15*
T0*
use_locking(*
_output_shapes

: *
validate_shape(*$
_class
loc:@pi/dense_1/kernel
�
save_1/Assign_16Assignpi/dense_2/biassave_1/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_1/Assign_17Assignpi/dense_2/bias/Adamsave_1/RestoreV2:17*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
�
save_1/Assign_18Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_1/Assign_19Assignpi/dense_2/kernelsave_1/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
�
save_1/Assign_20Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:20*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:*
validate_shape(
�
save_1/Assign_21Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:21*
validate_shape(*
T0*
_output_shapes

:*
use_locking(*$
_class
loc:@pi/dense_2/kernel
�
save_1/Assign_22Assignpi/dense_3/biassave_1/RestoreV2:22*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
use_locking(*
validate_shape(
�
save_1/Assign_23Assignpi/dense_3/bias/Adamsave_1/RestoreV2:23*
validate_shape(*
T0*"
_class
loc:@pi/dense_3/bias*
use_locking(*
_output_shapes
:
�
save_1/Assign_24Assignpi/dense_3/bias/Adam_1save_1/RestoreV2:24*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_3/bias
�
save_1/Assign_25Assignpi/dense_3/kernelsave_1/RestoreV2:25*
_output_shapes

:*$
_class
loc:@pi/dense_3/kernel*
T0*
use_locking(*
validate_shape(
�
save_1/Assign_26Assignpi/dense_3/kernel/Adamsave_1/RestoreV2:26*$
_class
loc:@pi/dense_3/kernel*
validate_shape(*
use_locking(*
_output_shapes

:*
T0
�
save_1/Assign_27Assignpi/dense_3/kernel/Adam_1save_1/RestoreV2:27*
T0*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
_output_shapes

:*
validate_shape(
�
save_1/Assign_28Assignv/dense/biassave_1/RestoreV2:28*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_1/Assign_29Assignv/dense/bias/Adamsave_1/RestoreV2:29*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*
_class
loc:@v/dense/bias
�
save_1/Assign_30Assignv/dense/bias/Adam_1save_1/RestoreV2:30*
use_locking(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
validate_shape(
�
save_1/Assign_31Assignv/dense/kernelsave_1/RestoreV2:31*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

: *
use_locking(*
T0
�
save_1/Assign_32Assignv/dense/kernel/Adamsave_1/RestoreV2:32*!
_class
loc:@v/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes

: *
T0
�
save_1/Assign_33Assignv/dense/kernel/Adam_1save_1/RestoreV2:33*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

: 
�
save_1/Assign_34Assignv/dense_1/biassave_1/RestoreV2:34*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_35Assignv/dense_1/bias/Adamsave_1/RestoreV2:35*
validate_shape(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_1/Assign_36Assignv/dense_1/bias/Adam_1save_1/RestoreV2:36*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
save_1/Assign_37Assignv/dense_1/kernelsave_1/RestoreV2:37*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

: 
�
save_1/Assign_38Assignv/dense_1/kernel/Adamsave_1/RestoreV2:38*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

: 
�
save_1/Assign_39Assignv/dense_1/kernel/Adam_1save_1/RestoreV2:39*
use_locking(*
T0*
_output_shapes

: *
validate_shape(*#
_class
loc:@v/dense_1/kernel
�
save_1/Assign_40Assignv/dense_2/biassave_1/RestoreV2:40*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0
�
save_1/Assign_41Assignv/dense_2/bias/Adamsave_1/RestoreV2:41*
use_locking(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_1/Assign_42Assignv/dense_2/bias/Adam_1save_1/RestoreV2:42*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias
�
save_1/Assign_43Assignv/dense_2/kernelsave_1/RestoreV2:43*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(
�
save_1/Assign_44Assignv/dense_2/kernel/Adamsave_1/RestoreV2:44*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
validate_shape(*
use_locking(*
T0
�
save_1/Assign_45Assignv/dense_2/kernel/Adam_1save_1/RestoreV2:45*
_output_shapes

:*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(
�
save_1/Assign_46Assignv/dense_3/biassave_1/RestoreV2:46*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
use_locking(*
T0
�
save_1/Assign_47Assignv/dense_3/bias/Adamsave_1/RestoreV2:47*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_3/bias
�
save_1/Assign_48Assignv/dense_3/bias/Adam_1save_1/RestoreV2:48*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
T0*
validate_shape(
�
save_1/Assign_49Assignv/dense_3/kernelsave_1/RestoreV2:49*
use_locking(*#
_class
loc:@v/dense_3/kernel*
T0*
_output_shapes

:*
validate_shape(
�
save_1/Assign_50Assignv/dense_3/kernel/Adamsave_1/RestoreV2:50*#
_class
loc:@v/dense_3/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_1/Assign_51Assignv/dense_3/kernel/Adam_1save_1/RestoreV2:51*#
_class
loc:@v/dense_3/kernel*
use_locking(*
T0*
_output_shapes

:*
validate_shape(
�
save_1/Assign_52Assignv/dense_4/biassave_1/RestoreV2:52*!
_class
loc:@v/dense_4/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_53Assignv/dense_4/bias/Adamsave_1/RestoreV2:53*
_output_shapes
:@*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_4/bias
�
save_1/Assign_54Assignv/dense_4/bias/Adam_1save_1/RestoreV2:54*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_4/bias
�
save_1/Assign_55Assignv/dense_4/kernelsave_1/RestoreV2:55*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
use_locking(*
validate_shape(*
T0
�
save_1/Assign_56Assignv/dense_4/kernel/Adamsave_1/RestoreV2:56*
_output_shapes
:	�@*
validate_shape(*
T0*#
_class
loc:@v/dense_4/kernel*
use_locking(
�
save_1/Assign_57Assignv/dense_4/kernel/Adam_1save_1/RestoreV2:57*
_output_shapes
:	�@*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_4/kernel
�
save_1/Assign_58Assignv/dense_5/biassave_1/RestoreV2:58*
T0*
_output_shapes
: *
validate_shape(*!
_class
loc:@v/dense_5/bias*
use_locking(
�
save_1/Assign_59Assignv/dense_5/bias/Adamsave_1/RestoreV2:59*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_5/bias
�
save_1/Assign_60Assignv/dense_5/bias/Adam_1save_1/RestoreV2:60*
T0*
use_locking(*!
_class
loc:@v/dense_5/bias*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_61Assignv/dense_5/kernelsave_1/RestoreV2:61*
T0*
_output_shapes

:@ *
validate_shape(*#
_class
loc:@v/dense_5/kernel*
use_locking(
�
save_1/Assign_62Assignv/dense_5/kernel/Adamsave_1/RestoreV2:62*#
_class
loc:@v/dense_5/kernel*
_output_shapes

:@ *
validate_shape(*
use_locking(*
T0
�
save_1/Assign_63Assignv/dense_5/kernel/Adam_1save_1/RestoreV2:63*
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_5/kernel*
_output_shapes

:@ 
�
save_1/Assign_64Assignv/dense_6/biassave_1/RestoreV2:64*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_6/bias*
T0*
use_locking(
�
save_1/Assign_65Assignv/dense_6/bias/Adamsave_1/RestoreV2:65*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_6/bias
�
save_1/Assign_66Assignv/dense_6/bias/Adam_1save_1/RestoreV2:66*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
use_locking(
�
save_1/Assign_67Assignv/dense_6/kernelsave_1/RestoreV2:67*
T0*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
validate_shape(*
use_locking(
�
save_1/Assign_68Assignv/dense_6/kernel/Adamsave_1/RestoreV2:68*
_output_shapes

: *
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_6/kernel
�
save_1/Assign_69Assignv/dense_6/kernel/Adam_1save_1/RestoreV2:69*
validate_shape(*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
T0*
use_locking(
�
save_1/Assign_70Assignv/dense_7/biassave_1/RestoreV2:70*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_7/bias
�
save_1/Assign_71Assignv/dense_7/bias/Adamsave_1/RestoreV2:71*
validate_shape(*!
_class
loc:@v/dense_7/bias*
T0*
use_locking(*
_output_shapes
:
�
save_1/Assign_72Assignv/dense_7/bias/Adam_1save_1/RestoreV2:72*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_7/bias
�
save_1/Assign_73Assignv/dense_7/kernelsave_1/RestoreV2:73*
T0*
_output_shapes

:*
validate_shape(*#
_class
loc:@v/dense_7/kernel*
use_locking(
�
save_1/Assign_74Assignv/dense_7/kernel/Adamsave_1/RestoreV2:74*
T0*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
use_locking(*
validate_shape(
�
save_1/Assign_75Assignv/dense_7/kernel/Adam_1save_1/RestoreV2:75*#
_class
loc:@v/dense_7/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:
�
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_b98df08747734b1a9de98b023ed4d35c/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
�
save_2/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
save_2/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_2/ShardedFilename
�
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*

axis *
_output_shapes
:*
N*
T0
�
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
�
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
�
save_2/RestoreV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0
�
!save_2/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L
�
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
_output_shapes
: 
�
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(
�
save_2/Assign_2Assignbeta2_powersave_2/RestoreV2:2*
T0*
use_locking(*
validate_shape(*
_output_shapes
: * 
_class
loc:@pi/dense/bias
�
save_2/Assign_3Assignbeta2_power_1save_2/RestoreV2:3*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
�
save_2/Assign_4Assignpi/dense/biassave_2/RestoreV2:4*
use_locking(*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_2/Assign_5Assignpi/dense/bias/Adamsave_2/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(
�
save_2/Assign_6Assignpi/dense/bias/Adam_1save_2/RestoreV2:6*
validate_shape(*
T0*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias
�
save_2/Assign_7Assignpi/dense/kernelsave_2/RestoreV2:7*
T0*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
�
save_2/Assign_8Assignpi/dense/kernel/Adamsave_2/RestoreV2:8*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
validate_shape(
�
save_2/Assign_9Assignpi/dense/kernel/Adam_1save_2/RestoreV2:9*
T0*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(
�
save_2/Assign_10Assignpi/dense_1/biassave_2/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_2/Assign_11Assignpi/dense_1/bias/Adamsave_2/RestoreV2:11*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(
�
save_2/Assign_12Assignpi/dense_1/bias/Adam_1save_2/RestoreV2:12*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(
�
save_2/Assign_13Assignpi/dense_1/kernelsave_2/RestoreV2:13*
use_locking(*
T0*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel*
validate_shape(
�
save_2/Assign_14Assignpi/dense_1/kernel/Adamsave_2/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

: *
use_locking(
�
save_2/Assign_15Assignpi/dense_1/kernel/Adam_1save_2/RestoreV2:15*
T0*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(
�
save_2/Assign_16Assignpi/dense_2/biassave_2/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_2/Assign_17Assignpi/dense_2/bias/Adamsave_2/RestoreV2:17*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
�
save_2/Assign_18Assignpi/dense_2/bias/Adam_1save_2/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_2/Assign_19Assignpi/dense_2/kernelsave_2/RestoreV2:19*
_output_shapes

:*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(
�
save_2/Assign_20Assignpi/dense_2/kernel/Adamsave_2/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:*
validate_shape(*
use_locking(*
T0
�
save_2/Assign_21Assignpi/dense_2/kernel/Adam_1save_2/RestoreV2:21*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:*
validate_shape(*
T0
�
save_2/Assign_22Assignpi/dense_3/biassave_2/RestoreV2:22*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
use_locking(*
T0
�
save_2/Assign_23Assignpi/dense_3/bias/Adamsave_2/RestoreV2:23*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
T0
�
save_2/Assign_24Assignpi/dense_3/bias/Adam_1save_2/RestoreV2:24*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias
�
save_2/Assign_25Assignpi/dense_3/kernelsave_2/RestoreV2:25*
_output_shapes

:*
use_locking(*
T0*$
_class
loc:@pi/dense_3/kernel*
validate_shape(
�
save_2/Assign_26Assignpi/dense_3/kernel/Adamsave_2/RestoreV2:26*
validate_shape(*
T0*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
_output_shapes

:
�
save_2/Assign_27Assignpi/dense_3/kernel/Adam_1save_2/RestoreV2:27*$
_class
loc:@pi/dense_3/kernel*
T0*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_2/Assign_28Assignv/dense/biassave_2/RestoreV2:28*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0
�
save_2/Assign_29Assignv/dense/bias/Adamsave_2/RestoreV2:29*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_2/Assign_30Assignv/dense/bias/Adam_1save_2/RestoreV2:30*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(
�
save_2/Assign_31Assignv/dense/kernelsave_2/RestoreV2:31*
_output_shapes

: *!
_class
loc:@v/dense/kernel*
use_locking(*
T0*
validate_shape(
�
save_2/Assign_32Assignv/dense/kernel/Adamsave_2/RestoreV2:32*
validate_shape(*
_output_shapes

: *!
_class
loc:@v/dense/kernel*
T0*
use_locking(
�
save_2/Assign_33Assignv/dense/kernel/Adam_1save_2/RestoreV2:33*
use_locking(*
_output_shapes

: *
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_2/Assign_34Assignv/dense_1/biassave_2/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0
�
save_2/Assign_35Assignv/dense_1/bias/Adamsave_2/RestoreV2:35*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
validate_shape(*
T0
�
save_2/Assign_36Assignv/dense_1/bias/Adam_1save_2/RestoreV2:36*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0*
validate_shape(
�
save_2/Assign_37Assignv/dense_1/kernelsave_2/RestoreV2:37*
use_locking(*
validate_shape(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_1/kernel
�
save_2/Assign_38Assignv/dense_1/kernel/Adamsave_2/RestoreV2:38*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0*
validate_shape(
�
save_2/Assign_39Assignv/dense_1/kernel/Adam_1save_2/RestoreV2:39*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
validate_shape(
�
save_2/Assign_40Assignv/dense_2/biassave_2/RestoreV2:40*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
�
save_2/Assign_41Assignv/dense_2/bias/Adamsave_2/RestoreV2:41*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_2/Assign_42Assignv/dense_2/bias/Adam_1save_2/RestoreV2:42*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(
�
save_2/Assign_43Assignv/dense_2/kernelsave_2/RestoreV2:43*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:
�
save_2/Assign_44Assignv/dense_2/kernel/Adamsave_2/RestoreV2:44*
_output_shapes

:*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_2/Assign_45Assignv/dense_2/kernel/Adam_1save_2/RestoreV2:45*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(*
use_locking(
�
save_2/Assign_46Assignv/dense_3/biassave_2/RestoreV2:46*!
_class
loc:@v/dense_3/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_2/Assign_47Assignv/dense_3/bias/Adamsave_2/RestoreV2:47*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias
�
save_2/Assign_48Assignv/dense_3/bias/Adam_1save_2/RestoreV2:48*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
validate_shape(*
T0*
use_locking(
�
save_2/Assign_49Assignv/dense_3/kernelsave_2/RestoreV2:49*#
_class
loc:@v/dense_3/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_2/Assign_50Assignv/dense_3/kernel/Adamsave_2/RestoreV2:50*
_output_shapes

:*
T0*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
use_locking(
�
save_2/Assign_51Assignv/dense_3/kernel/Adam_1save_2/RestoreV2:51*#
_class
loc:@v/dense_3/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
�
save_2/Assign_52Assignv/dense_4/biassave_2/RestoreV2:52*
use_locking(*!
_class
loc:@v/dense_4/bias*
_output_shapes
:@*
validate_shape(*
T0
�
save_2/Assign_53Assignv/dense_4/bias/Adamsave_2/RestoreV2:53*
T0*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_4/bias*
validate_shape(
�
save_2/Assign_54Assignv/dense_4/bias/Adam_1save_2/RestoreV2:54*
validate_shape(*!
_class
loc:@v/dense_4/bias*
T0*
_output_shapes
:@*
use_locking(
�
save_2/Assign_55Assignv/dense_4/kernelsave_2/RestoreV2:55*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
use_locking(*
T0*
_output_shapes
:	�@
�
save_2/Assign_56Assignv/dense_4/kernel/Adamsave_2/RestoreV2:56*
T0*
use_locking(*#
_class
loc:@v/dense_4/kernel*
validate_shape(*
_output_shapes
:	�@
�
save_2/Assign_57Assignv/dense_4/kernel/Adam_1save_2/RestoreV2:57*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel
�
save_2/Assign_58Assignv/dense_5/biassave_2/RestoreV2:58*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@v/dense_5/bias
�
save_2/Assign_59Assignv/dense_5/bias/Adamsave_2/RestoreV2:59*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
T0
�
save_2/Assign_60Assignv/dense_5/bias/Adam_1save_2/RestoreV2:60*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
validate_shape(*
T0*
use_locking(
�
save_2/Assign_61Assignv/dense_5/kernelsave_2/RestoreV2:61*
_output_shapes

:@ *
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_5/kernel
�
save_2/Assign_62Assignv/dense_5/kernel/Adamsave_2/RestoreV2:62*
validate_shape(*#
_class
loc:@v/dense_5/kernel*
T0*
use_locking(*
_output_shapes

:@ 
�
save_2/Assign_63Assignv/dense_5/kernel/Adam_1save_2/RestoreV2:63*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel
�
save_2/Assign_64Assignv/dense_6/biassave_2/RestoreV2:64*
use_locking(*!
_class
loc:@v/dense_6/bias*
T0*
_output_shapes
:*
validate_shape(
�
save_2/Assign_65Assignv/dense_6/bias/Adamsave_2/RestoreV2:65*!
_class
loc:@v/dense_6/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_2/Assign_66Assignv/dense_6/bias/Adam_1save_2/RestoreV2:66*!
_class
loc:@v/dense_6/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_2/Assign_67Assignv/dense_6/kernelsave_2/RestoreV2:67*
use_locking(*
T0*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
validate_shape(
�
save_2/Assign_68Assignv/dense_6/kernel/Adamsave_2/RestoreV2:68*
_output_shapes

: *
use_locking(*#
_class
loc:@v/dense_6/kernel*
validate_shape(*
T0
�
save_2/Assign_69Assignv/dense_6/kernel/Adam_1save_2/RestoreV2:69*
_output_shapes

: *
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_6/kernel
�
save_2/Assign_70Assignv/dense_7/biassave_2/RestoreV2:70*!
_class
loc:@v/dense_7/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_2/Assign_71Assignv/dense_7/bias/Adamsave_2/RestoreV2:71*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_7/bias*
_output_shapes
:
�
save_2/Assign_72Assignv/dense_7/bias/Adam_1save_2/RestoreV2:72*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_7/bias
�
save_2/Assign_73Assignv/dense_7/kernelsave_2/RestoreV2:73*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@v/dense_7/kernel*
validate_shape(
�
save_2/Assign_74Assignv/dense_7/kernel/Adamsave_2/RestoreV2:74*
use_locking(*
_output_shapes

:*
T0*
validate_shape(*#
_class
loc:@v/dense_7/kernel
�
save_2/Assign_75Assignv/dense_7/kernel/Adam_1save_2/RestoreV2:75*
validate_shape(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
use_locking(
�
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_5^save_2/Assign_50^save_2/Assign_51^save_2/Assign_52^save_2/Assign_53^save_2/Assign_54^save_2/Assign_55^save_2/Assign_56^save_2/Assign_57^save_2/Assign_58^save_2/Assign_59^save_2/Assign_6^save_2/Assign_60^save_2/Assign_61^save_2/Assign_62^save_2/Assign_63^save_2/Assign_64^save_2/Assign_65^save_2/Assign_66^save_2/Assign_67^save_2/Assign_68^save_2/Assign_69^save_2/Assign_7^save_2/Assign_70^save_2/Assign_71^save_2/Assign_72^save_2/Assign_73^save_2/Assign_74^save_2/Assign_75^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
_output_shapes
: *
shape: *
dtype0
�
save_3/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_1a642b4823f84eb4b42f0370aacbc1ea/part
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_3/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
�
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
�
save_3/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
_output_shapes
:L*
dtype0
�
save_3/SaveV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: *
T0
�
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
T0*
_output_shapes
:*

axis *
N
�
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
�
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
�
save_3/RestoreV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
!save_3/RestoreV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*
use_locking(*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
�
save_3/Assign_2Assignbeta2_powersave_3/RestoreV2:2* 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
�
save_3/Assign_3Assignbeta2_power_1save_3/RestoreV2:3*
_output_shapes
: *
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
T0
�
save_3/Assign_4Assignpi/dense/biassave_3/RestoreV2:4* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
�
save_3/Assign_5Assignpi/dense/bias/Adamsave_3/RestoreV2:5* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
�
save_3/Assign_6Assignpi/dense/bias/Adam_1save_3/RestoreV2:6* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
�
save_3/Assign_7Assignpi/dense/kernelsave_3/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

: 
�
save_3/Assign_8Assignpi/dense/kernel/Adamsave_3/RestoreV2:8*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
validate_shape(*
T0
�
save_3/Assign_9Assignpi/dense/kernel/Adam_1save_3/RestoreV2:9*
T0*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
use_locking(*
validate_shape(
�
save_3/Assign_10Assignpi/dense_1/biassave_3/RestoreV2:10*
validate_shape(*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:
�
save_3/Assign_11Assignpi/dense_1/bias/Adamsave_3/RestoreV2:11*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
_output_shapes
:
�
save_3/Assign_12Assignpi/dense_1/bias/Adam_1save_3/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_3/Assign_13Assignpi/dense_1/kernelsave_3/RestoreV2:13*
T0*
validate_shape(*
_output_shapes

: *
use_locking(*$
_class
loc:@pi/dense_1/kernel
�
save_3/Assign_14Assignpi/dense_1/kernel/Adamsave_3/RestoreV2:14*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: *
use_locking(*
T0
�
save_3/Assign_15Assignpi/dense_1/kernel/Adam_1save_3/RestoreV2:15*
validate_shape(*
use_locking(*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel*
T0
�
save_3/Assign_16Assignpi/dense_2/biassave_3/RestoreV2:16*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(
�
save_3/Assign_17Assignpi/dense_2/bias/Adamsave_3/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_3/Assign_18Assignpi/dense_2/bias/Adam_1save_3/RestoreV2:18*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias
�
save_3/Assign_19Assignpi/dense_2/kernelsave_3/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:*
use_locking(*
validate_shape(*
T0
�
save_3/Assign_20Assignpi/dense_2/kernel/Adamsave_3/RestoreV2:20*
use_locking(*
_output_shapes

:*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_3/Assign_21Assignpi/dense_2/kernel/Adam_1save_3/RestoreV2:21*
validate_shape(*
use_locking(*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
T0
�
save_3/Assign_22Assignpi/dense_3/biassave_3/RestoreV2:22*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_3/bias
�
save_3/Assign_23Assignpi/dense_3/bias/Adamsave_3/RestoreV2:23*
T0*"
_class
loc:@pi/dense_3/bias*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_3/Assign_24Assignpi/dense_3/bias/Adam_1save_3/RestoreV2:24*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_3/bias
�
save_3/Assign_25Assignpi/dense_3/kernelsave_3/RestoreV2:25*
T0*
use_locking(*$
_class
loc:@pi/dense_3/kernel*
validate_shape(*
_output_shapes

:
�
save_3/Assign_26Assignpi/dense_3/kernel/Adamsave_3/RestoreV2:26*
use_locking(*
T0*
_output_shapes

:*
validate_shape(*$
_class
loc:@pi/dense_3/kernel
�
save_3/Assign_27Assignpi/dense_3/kernel/Adam_1save_3/RestoreV2:27*
use_locking(*
_output_shapes

:*
T0*
validate_shape(*$
_class
loc:@pi/dense_3/kernel
�
save_3/Assign_28Assignv/dense/biassave_3/RestoreV2:28*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(*
T0
�
save_3/Assign_29Assignv/dense/bias/Adamsave_3/RestoreV2:29*
T0*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_30Assignv/dense/bias/Adam_1save_3/RestoreV2:30*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
save_3/Assign_31Assignv/dense/kernelsave_3/RestoreV2:31*
use_locking(*
_output_shapes

: *
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_3/Assign_32Assignv/dense/kernel/Adamsave_3/RestoreV2:32*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

: *
use_locking(*
validate_shape(
�
save_3/Assign_33Assignv/dense/kernel/Adam_1save_3/RestoreV2:33*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

: *
use_locking(
�
save_3/Assign_34Assignv/dense_1/biassave_3/RestoreV2:34*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias
�
save_3/Assign_35Assignv/dense_1/bias/Adamsave_3/RestoreV2:35*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_3/Assign_36Assignv/dense_1/bias/Adam_1save_3/RestoreV2:36*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
�
save_3/Assign_37Assignv/dense_1/kernelsave_3/RestoreV2:37*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
validate_shape(*
T0*
use_locking(
�
save_3/Assign_38Assignv/dense_1/kernel/Adamsave_3/RestoreV2:38*
use_locking(*
_output_shapes

: *
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0
�
save_3/Assign_39Assignv/dense_1/kernel/Adam_1save_3/RestoreV2:39*
_output_shapes

: *
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_3/Assign_40Assignv/dense_2/biassave_3/RestoreV2:40*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
�
save_3/Assign_41Assignv/dense_2/bias/Adamsave_3/RestoreV2:41*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
T0
�
save_3/Assign_42Assignv/dense_2/bias/Adam_1save_3/RestoreV2:42*
use_locking(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_3/Assign_43Assignv/dense_2/kernelsave_3/RestoreV2:43*
validate_shape(*
use_locking(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel
�
save_3/Assign_44Assignv/dense_2/kernel/Adamsave_3/RestoreV2:44*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:*
use_locking(
�
save_3/Assign_45Assignv/dense_2/kernel/Adam_1save_3/RestoreV2:45*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0
�
save_3/Assign_46Assignv/dense_3/biassave_3/RestoreV2:46*
validate_shape(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_3/bias*
use_locking(
�
save_3/Assign_47Assignv/dense_3/bias/Adamsave_3/RestoreV2:47*!
_class
loc:@v/dense_3/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
�
save_3/Assign_48Assignv/dense_3/bias/Adam_1save_3/RestoreV2:48*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_3/Assign_49Assignv/dense_3/kernelsave_3/RestoreV2:49*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:
�
save_3/Assign_50Assignv/dense_3/kernel/Adamsave_3/RestoreV2:50*
_output_shapes

:*
T0*#
_class
loc:@v/dense_3/kernel*
use_locking(*
validate_shape(
�
save_3/Assign_51Assignv/dense_3/kernel/Adam_1save_3/RestoreV2:51*
validate_shape(*
T0*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
use_locking(
�
save_3/Assign_52Assignv/dense_4/biassave_3/RestoreV2:52*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_4/bias*
validate_shape(*
T0
�
save_3/Assign_53Assignv/dense_4/bias/Adamsave_3/RestoreV2:53*
validate_shape(*!
_class
loc:@v/dense_4/bias*
T0*
use_locking(*
_output_shapes
:@
�
save_3/Assign_54Assignv/dense_4/bias/Adam_1save_3/RestoreV2:54*
validate_shape(*!
_class
loc:@v/dense_4/bias*
use_locking(*
_output_shapes
:@*
T0
�
save_3/Assign_55Assignv/dense_4/kernelsave_3/RestoreV2:55*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*
validate_shape(*
use_locking(*
T0
�
save_3/Assign_56Assignv/dense_4/kernel/Adamsave_3/RestoreV2:56*
_output_shapes
:	�@*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
T0*
use_locking(
�
save_3/Assign_57Assignv/dense_4/kernel/Adam_1save_3/RestoreV2:57*
validate_shape(*
T0*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*
use_locking(
�
save_3/Assign_58Assignv/dense_5/biassave_3/RestoreV2:58*!
_class
loc:@v/dense_5/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
�
save_3/Assign_59Assignv/dense_5/bias/Adamsave_3/RestoreV2:59*
_output_shapes
: *
validate_shape(*!
_class
loc:@v/dense_5/bias*
T0*
use_locking(
�
save_3/Assign_60Assignv/dense_5/bias/Adam_1save_3/RestoreV2:60*
T0*
use_locking(*!
_class
loc:@v/dense_5/bias*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_61Assignv/dense_5/kernelsave_3/RestoreV2:61*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_5/kernel*
T0*
_output_shapes

:@ 
�
save_3/Assign_62Assignv/dense_5/kernel/Adamsave_3/RestoreV2:62*
validate_shape(*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel*
use_locking(*
T0
�
save_3/Assign_63Assignv/dense_5/kernel/Adam_1save_3/RestoreV2:63*
T0*
validate_shape(*
use_locking(*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel
�
save_3/Assign_64Assignv/dense_6/biassave_3/RestoreV2:64*!
_class
loc:@v/dense_6/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
save_3/Assign_65Assignv/dense_6/bias/Adamsave_3/RestoreV2:65*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_6/bias*
use_locking(*
T0
�
save_3/Assign_66Assignv/dense_6/bias/Adam_1save_3/RestoreV2:66*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
T0
�
save_3/Assign_67Assignv/dense_6/kernelsave_3/RestoreV2:67*
validate_shape(*
_output_shapes

: *
T0*
use_locking(*#
_class
loc:@v/dense_6/kernel
�
save_3/Assign_68Assignv/dense_6/kernel/Adamsave_3/RestoreV2:68*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
use_locking(*
validate_shape(*
T0
�
save_3/Assign_69Assignv/dense_6/kernel/Adam_1save_3/RestoreV2:69*#
_class
loc:@v/dense_6/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

: 
�
save_3/Assign_70Assignv/dense_7/biassave_3/RestoreV2:70*
validate_shape(*!
_class
loc:@v/dense_7/bias*
_output_shapes
:*
use_locking(*
T0
�
save_3/Assign_71Assignv/dense_7/bias/Adamsave_3/RestoreV2:71*
T0*
validate_shape(*!
_class
loc:@v/dense_7/bias*
use_locking(*
_output_shapes
:
�
save_3/Assign_72Assignv/dense_7/bias/Adam_1save_3/RestoreV2:72*!
_class
loc:@v/dense_7/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
�
save_3/Assign_73Assignv/dense_7/kernelsave_3/RestoreV2:73*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
use_locking(*
_output_shapes

:*
T0
�
save_3/Assign_74Assignv/dense_7/kernel/Adamsave_3/RestoreV2:74*
validate_shape(*#
_class
loc:@v/dense_7/kernel*
T0*
_output_shapes

:*
use_locking(
�
save_3/Assign_75Assignv/dense_7/kernel/Adam_1save_3/RestoreV2:75*
_output_shapes

:*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_7/kernel
�
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_51^save_3/Assign_52^save_3/Assign_53^save_3/Assign_54^save_3/Assign_55^save_3/Assign_56^save_3/Assign_57^save_3/Assign_58^save_3/Assign_59^save_3/Assign_6^save_3/Assign_60^save_3/Assign_61^save_3/Assign_62^save_3/Assign_63^save_3/Assign_64^save_3/Assign_65^save_3/Assign_66^save_3/Assign_67^save_3/Assign_68^save_3/Assign_69^save_3/Assign_7^save_3/Assign_70^save_3/Assign_71^save_3/Assign_72^save_3/Assign_73^save_3/Assign_74^save_3/Assign_75^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_4/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_f1b24db14d384592857b45e7f6829c8b/part
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_4/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_4/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
�
save_4/SaveV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
save_4/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 
�
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
T0*
_output_shapes
:*
N*

axis 
�
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
�
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
_output_shapes
: *
T0
�
save_4/RestoreV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0
�
!save_4/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(
�
save_4/Assign_1Assignbeta1_power_1save_4/RestoreV2:1*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(
�
save_4/Assign_2Assignbeta2_powersave_4/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
�
save_4/Assign_3Assignbeta2_power_1save_4/RestoreV2:3*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias
�
save_4/Assign_4Assignpi/dense/biassave_4/RestoreV2:4* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
�
save_4/Assign_5Assignpi/dense/bias/Adamsave_4/RestoreV2:5* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
�
save_4/Assign_6Assignpi/dense/bias/Adam_1save_4/RestoreV2:6*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
validate_shape(
�
save_4/Assign_7Assignpi/dense/kernelsave_4/RestoreV2:7*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

: *
T0
�
save_4/Assign_8Assignpi/dense/kernel/Adamsave_4/RestoreV2:8*
T0*
_output_shapes

: *
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(
�
save_4/Assign_9Assignpi/dense/kernel/Adam_1save_4/RestoreV2:9*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
validate_shape(
�
save_4/Assign_10Assignpi/dense_1/biassave_4/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
save_4/Assign_11Assignpi/dense_1/bias/Adamsave_4/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
:
�
save_4/Assign_12Assignpi/dense_1/bias/Adam_1save_4/RestoreV2:12*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_4/Assign_13Assignpi/dense_1/kernelsave_4/RestoreV2:13*
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
_output_shapes

: 
�
save_4/Assign_14Assignpi/dense_1/kernel/Adamsave_4/RestoreV2:14*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
use_locking(*
_output_shapes

: 
�
save_4/Assign_15Assignpi/dense_1/kernel/Adam_1save_4/RestoreV2:15*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: 
�
save_4/Assign_16Assignpi/dense_2/biassave_4/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
save_4/Assign_17Assignpi/dense_2/bias/Adamsave_4/RestoreV2:17*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_4/Assign_18Assignpi/dense_2/bias/Adam_1save_4/RestoreV2:18*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0
�
save_4/Assign_19Assignpi/dense_2/kernelsave_4/RestoreV2:19*
use_locking(*
_output_shapes

:*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel
�
save_4/Assign_20Assignpi/dense_2/kernel/Adamsave_4/RestoreV2:20*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:
�
save_4/Assign_21Assignpi/dense_2/kernel/Adam_1save_4/RestoreV2:21*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:
�
save_4/Assign_22Assignpi/dense_3/biassave_4/RestoreV2:22*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
use_locking(*
validate_shape(
�
save_4/Assign_23Assignpi/dense_3/bias/Adamsave_4/RestoreV2:23*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias
�
save_4/Assign_24Assignpi/dense_3/bias/Adam_1save_4/RestoreV2:24*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_3/bias
�
save_4/Assign_25Assignpi/dense_3/kernelsave_4/RestoreV2:25*
use_locking(*
T0*$
_class
loc:@pi/dense_3/kernel*
validate_shape(*
_output_shapes

:
�
save_4/Assign_26Assignpi/dense_3/kernel/Adamsave_4/RestoreV2:26*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
T0*
validate_shape(*
use_locking(
�
save_4/Assign_27Assignpi/dense_3/kernel/Adam_1save_4/RestoreV2:27*
_output_shapes

:*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_3/kernel
�
save_4/Assign_28Assignv/dense/biassave_4/RestoreV2:28*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
�
save_4/Assign_29Assignv/dense/bias/Adamsave_4/RestoreV2:29*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
: 
�
save_4/Assign_30Assignv/dense/bias/Adam_1save_4/RestoreV2:30*
T0*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
�
save_4/Assign_31Assignv/dense/kernelsave_4/RestoreV2:31*
use_locking(*
_output_shapes

: *!
_class
loc:@v/dense/kernel*
validate_shape(*
T0
�
save_4/Assign_32Assignv/dense/kernel/Adamsave_4/RestoreV2:32*!
_class
loc:@v/dense/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

: 
�
save_4/Assign_33Assignv/dense/kernel/Adam_1save_4/RestoreV2:33*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

: *
T0*
use_locking(
�
save_4/Assign_34Assignv/dense_1/biassave_4/RestoreV2:34*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:
�
save_4/Assign_35Assignv/dense_1/bias/Adamsave_4/RestoreV2:35*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_4/Assign_36Assignv/dense_1/bias/Adam_1save_4/RestoreV2:36*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_4/Assign_37Assignv/dense_1/kernelsave_4/RestoreV2:37*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
use_locking(*
validate_shape(*
T0
�
save_4/Assign_38Assignv/dense_1/kernel/Adamsave_4/RestoreV2:38*
use_locking(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
�
save_4/Assign_39Assignv/dense_1/kernel/Adam_1save_4/RestoreV2:39*
validate_shape(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_4/Assign_40Assignv/dense_2/biassave_4/RestoreV2:40*
_output_shapes
:*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0
�
save_4/Assign_41Assignv/dense_2/bias/Adamsave_4/RestoreV2:41*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
�
save_4/Assign_42Assignv/dense_2/bias/Adam_1save_4/RestoreV2:42*!
_class
loc:@v/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_4/Assign_43Assignv/dense_2/kernelsave_4/RestoreV2:43*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_4/Assign_44Assignv/dense_2/kernel/Adamsave_4/RestoreV2:44*
_output_shapes

:*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_4/Assign_45Assignv/dense_2/kernel/Adam_1save_4/RestoreV2:45*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(*
_output_shapes

:*
use_locking(
�
save_4/Assign_46Assignv/dense_3/biassave_4/RestoreV2:46*!
_class
loc:@v/dense_3/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
�
save_4/Assign_47Assignv/dense_3/bias/Adamsave_4/RestoreV2:47*
T0*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
use_locking(*
validate_shape(
�
save_4/Assign_48Assignv/dense_3/bias/Adam_1save_4/RestoreV2:48*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
T0
�
save_4/Assign_49Assignv/dense_3/kernelsave_4/RestoreV2:49*
use_locking(*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
_output_shapes

:*
T0
�
save_4/Assign_50Assignv/dense_3/kernel/Adamsave_4/RestoreV2:50*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
validate_shape(*
use_locking(*
T0
�
save_4/Assign_51Assignv/dense_3/kernel/Adam_1save_4/RestoreV2:51*
use_locking(*
T0*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel
�
save_4/Assign_52Assignv/dense_4/biassave_4/RestoreV2:52*
validate_shape(*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
T0
�
save_4/Assign_53Assignv/dense_4/bias/Adamsave_4/RestoreV2:53*
use_locking(*!
_class
loc:@v/dense_4/bias*
_output_shapes
:@*
validate_shape(*
T0
�
save_4/Assign_54Assignv/dense_4/bias/Adam_1save_4/RestoreV2:54*
T0*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_4/bias*
validate_shape(
�
save_4/Assign_55Assignv/dense_4/kernelsave_4/RestoreV2:55*#
_class
loc:@v/dense_4/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	�@*
T0
�
save_4/Assign_56Assignv/dense_4/kernel/Adamsave_4/RestoreV2:56*
_output_shapes
:	�@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
T0
�
save_4/Assign_57Assignv/dense_4/kernel/Adam_1save_4/RestoreV2:57*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*
T0*
use_locking(
�
save_4/Assign_58Assignv/dense_5/biassave_4/RestoreV2:58*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_5/bias
�
save_4/Assign_59Assignv/dense_5/bias/Adamsave_4/RestoreV2:59*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@v/dense_5/bias
�
save_4/Assign_60Assignv/dense_5/bias/Adam_1save_4/RestoreV2:60*
_output_shapes
: *
T0*
use_locking(*!
_class
loc:@v/dense_5/bias*
validate_shape(
�
save_4/Assign_61Assignv/dense_5/kernelsave_4/RestoreV2:61*
validate_shape(*#
_class
loc:@v/dense_5/kernel*
_output_shapes

:@ *
use_locking(*
T0
�
save_4/Assign_62Assignv/dense_5/kernel/Adamsave_4/RestoreV2:62*
validate_shape(*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel*
use_locking(*
T0
�
save_4/Assign_63Assignv/dense_5/kernel/Adam_1save_4/RestoreV2:63*
_output_shapes

:@ *
T0*
validate_shape(*#
_class
loc:@v/dense_5/kernel*
use_locking(
�
save_4/Assign_64Assignv/dense_6/biassave_4/RestoreV2:64*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
use_locking(*
T0
�
save_4/Assign_65Assignv/dense_6/bias/Adamsave_4/RestoreV2:65*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_6/bias
�
save_4/Assign_66Assignv/dense_6/bias/Adam_1save_4/RestoreV2:66*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_6/bias
�
save_4/Assign_67Assignv/dense_6/kernelsave_4/RestoreV2:67*
validate_shape(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_6/kernel*
use_locking(
�
save_4/Assign_68Assignv/dense_6/kernel/Adamsave_4/RestoreV2:68*
use_locking(*
_output_shapes

: *
T0*
validate_shape(*#
_class
loc:@v/dense_6/kernel
�
save_4/Assign_69Assignv/dense_6/kernel/Adam_1save_4/RestoreV2:69*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
T0*
validate_shape(*
use_locking(
�
save_4/Assign_70Assignv/dense_7/biassave_4/RestoreV2:70*
_output_shapes
:*!
_class
loc:@v/dense_7/bias*
T0*
validate_shape(*
use_locking(
�
save_4/Assign_71Assignv/dense_7/bias/Adamsave_4/RestoreV2:71*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_7/bias*
T0*
validate_shape(
�
save_4/Assign_72Assignv/dense_7/bias/Adam_1save_4/RestoreV2:72*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_7/bias*
_output_shapes
:
�
save_4/Assign_73Assignv/dense_7/kernelsave_4/RestoreV2:73*
validate_shape(*#
_class
loc:@v/dense_7/kernel*
use_locking(*
T0*
_output_shapes

:
�
save_4/Assign_74Assignv/dense_7/kernel/Adamsave_4/RestoreV2:74*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
_output_shapes

:*
T0*
use_locking(
�
save_4/Assign_75Assignv/dense_7/kernel/Adam_1save_4/RestoreV2:75*
validate_shape(*
T0*#
_class
loc:@v/dense_7/kernel*
_output_shapes

:*
use_locking(
�
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_45^save_4/Assign_46^save_4/Assign_47^save_4/Assign_48^save_4/Assign_49^save_4/Assign_5^save_4/Assign_50^save_4/Assign_51^save_4/Assign_52^save_4/Assign_53^save_4/Assign_54^save_4/Assign_55^save_4/Assign_56^save_4/Assign_57^save_4/Assign_58^save_4/Assign_59^save_4/Assign_6^save_4/Assign_60^save_4/Assign_61^save_4/Assign_62^save_4/Assign_63^save_4/Assign_64^save_4/Assign_65^save_4/Assign_66^save_4/Assign_67^save_4/Assign_68^save_4/Assign_69^save_4/Assign_7^save_4/Assign_70^save_4/Assign_71^save_4/Assign_72^save_4/Assign_73^save_4/Assign_74^save_4/Assign_75^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
_output_shapes
: *
dtype0*
shape: 
�
save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_0f27c742f89d4e4490864fa0255b1104/part*
_output_shapes
: *
dtype0
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_5/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_5/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
�
save_5/SaveV2/tensor_namesConst*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
_output_shapes
:L
�
save_5/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*)
_class
loc:@save_5/ShardedFilename*
T0*
_output_shapes
: 
�
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
_output_shapes
:*
N*

axis *
T0
�
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
�
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
�
save_5/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
!save_5/RestoreV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(*
_output_shapes
: 
�
save_5/Assign_1Assignbeta1_power_1save_5/RestoreV2:1*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: *
T0
�
save_5/Assign_2Assignbeta2_powersave_5/RestoreV2:2*
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
�
save_5/Assign_3Assignbeta2_power_1save_5/RestoreV2:3*
use_locking(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias*
validate_shape(
�
save_5/Assign_4Assignpi/dense/biassave_5/RestoreV2:4*
_output_shapes
: *
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_5/Assign_5Assignpi/dense/bias/Adamsave_5/RestoreV2:5*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0
�
save_5/Assign_6Assignpi/dense/bias/Adam_1save_5/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_5/Assign_7Assignpi/dense/kernelsave_5/RestoreV2:7*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

: *
T0
�
save_5/Assign_8Assignpi/dense/kernel/Adamsave_5/RestoreV2:8*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(*
T0*
_output_shapes

: 
�
save_5/Assign_9Assignpi/dense/kernel/Adam_1save_5/RestoreV2:9*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*"
_class
loc:@pi/dense/kernel
�
save_5/Assign_10Assignpi/dense_1/biassave_5/RestoreV2:10*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
save_5/Assign_11Assignpi/dense_1/bias/Adamsave_5/RestoreV2:11*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:
�
save_5/Assign_12Assignpi/dense_1/bias/Adam_1save_5/RestoreV2:12*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(
�
save_5/Assign_13Assignpi/dense_1/kernelsave_5/RestoreV2:13*
_output_shapes

: *
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel
�
save_5/Assign_14Assignpi/dense_1/kernel/Adamsave_5/RestoreV2:14*
validate_shape(*
use_locking(*
_output_shapes

: *
T0*$
_class
loc:@pi/dense_1/kernel
�
save_5/Assign_15Assignpi/dense_1/kernel/Adam_1save_5/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

: 
�
save_5/Assign_16Assignpi/dense_2/biassave_5/RestoreV2:16*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_5/Assign_17Assignpi/dense_2/bias/Adamsave_5/RestoreV2:17*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(
�
save_5/Assign_18Assignpi/dense_2/bias/Adam_1save_5/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_19Assignpi/dense_2/kernelsave_5/RestoreV2:19*
_output_shapes

:*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(
�
save_5/Assign_20Assignpi/dense_2/kernel/Adamsave_5/RestoreV2:20*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
save_5/Assign_21Assignpi/dense_2/kernel/Adam_1save_5/RestoreV2:21*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes

:
�
save_5/Assign_22Assignpi/dense_3/biassave_5/RestoreV2:22*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_3/bias
�
save_5/Assign_23Assignpi/dense_3/bias/Adamsave_5/RestoreV2:23*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
use_locking(*
_output_shapes
:*
T0
�
save_5/Assign_24Assignpi/dense_3/bias/Adam_1save_5/RestoreV2:24*"
_class
loc:@pi/dense_3/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
�
save_5/Assign_25Assignpi/dense_3/kernelsave_5/RestoreV2:25*$
_class
loc:@pi/dense_3/kernel*
T0*
use_locking(*
_output_shapes

:*
validate_shape(
�
save_5/Assign_26Assignpi/dense_3/kernel/Adamsave_5/RestoreV2:26*
use_locking(*$
_class
loc:@pi/dense_3/kernel*
validate_shape(*
_output_shapes

:*
T0
�
save_5/Assign_27Assignpi/dense_3/kernel/Adam_1save_5/RestoreV2:27*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0*
use_locking(
�
save_5/Assign_28Assignv/dense/biassave_5/RestoreV2:28*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
�
save_5/Assign_29Assignv/dense/bias/Adamsave_5/RestoreV2:29*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias
�
save_5/Assign_30Assignv/dense/bias/Adam_1save_5/RestoreV2:30*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
�
save_5/Assign_31Assignv/dense/kernelsave_5/RestoreV2:31*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

: *
T0
�
save_5/Assign_32Assignv/dense/kernel/Adamsave_5/RestoreV2:32*
T0*
use_locking(*
_output_shapes

: *
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_5/Assign_33Assignv/dense/kernel/Adam_1save_5/RestoreV2:33*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

: *
validate_shape(
�
save_5/Assign_34Assignv/dense_1/biassave_5/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
�
save_5/Assign_35Assignv/dense_1/bias/Adamsave_5/RestoreV2:35*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:
�
save_5/Assign_36Assignv/dense_1/bias/Adam_1save_5/RestoreV2:36*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
save_5/Assign_37Assignv/dense_1/kernelsave_5/RestoreV2:37*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

: *
use_locking(
�
save_5/Assign_38Assignv/dense_1/kernel/Adamsave_5/RestoreV2:38*
_output_shapes

: *
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0
�
save_5/Assign_39Assignv/dense_1/kernel/Adam_1save_5/RestoreV2:39*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

: *
validate_shape(*
use_locking(
�
save_5/Assign_40Assignv/dense_2/biassave_5/RestoreV2:40*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:
�
save_5/Assign_41Assignv/dense_2/bias/Adamsave_5/RestoreV2:41*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
�
save_5/Assign_42Assignv/dense_2/bias/Adam_1save_5/RestoreV2:42*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias
�
save_5/Assign_43Assignv/dense_2/kernelsave_5/RestoreV2:43*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
T0
�
save_5/Assign_44Assignv/dense_2/kernel/Adamsave_5/RestoreV2:44*
T0*
_output_shapes

:*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_5/Assign_45Assignv/dense_2/kernel/Adam_1save_5/RestoreV2:45*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:*
T0*
use_locking(
�
save_5/Assign_46Assignv/dense_3/biassave_5/RestoreV2:46*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
save_5/Assign_47Assignv/dense_3/bias/Adamsave_5/RestoreV2:47*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_3/bias*
validate_shape(
�
save_5/Assign_48Assignv/dense_3/bias/Adam_1save_5/RestoreV2:48*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_5/Assign_49Assignv/dense_3/kernelsave_5/RestoreV2:49*
_output_shapes

:*
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_3/kernel
�
save_5/Assign_50Assignv/dense_3/kernel/Adamsave_5/RestoreV2:50*
_output_shapes

:*
T0*
use_locking(*#
_class
loc:@v/dense_3/kernel*
validate_shape(
�
save_5/Assign_51Assignv/dense_3/kernel/Adam_1save_5/RestoreV2:51*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
use_locking(*
T0
�
save_5/Assign_52Assignv/dense_4/biassave_5/RestoreV2:52*!
_class
loc:@v/dense_4/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:@
�
save_5/Assign_53Assignv/dense_4/bias/Adamsave_5/RestoreV2:53*
_output_shapes
:@*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_4/bias*
T0
�
save_5/Assign_54Assignv/dense_4/bias/Adam_1save_5/RestoreV2:54*
T0*!
_class
loc:@v/dense_4/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_5/Assign_55Assignv/dense_4/kernelsave_5/RestoreV2:55*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@
�
save_5/Assign_56Assignv/dense_4/kernel/Adamsave_5/RestoreV2:56*
use_locking(*
T0*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
validate_shape(
�
save_5/Assign_57Assignv/dense_4/kernel/Adam_1save_5/RestoreV2:57*
_output_shapes
:	�@*
T0*
use_locking(*#
_class
loc:@v/dense_4/kernel*
validate_shape(
�
save_5/Assign_58Assignv/dense_5/biassave_5/RestoreV2:58*
validate_shape(*
_output_shapes
: *
T0*!
_class
loc:@v/dense_5/bias*
use_locking(
�
save_5/Assign_59Assignv/dense_5/bias/Adamsave_5/RestoreV2:59*
T0*
use_locking(*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
validate_shape(
�
save_5/Assign_60Assignv/dense_5/bias/Adam_1save_5/RestoreV2:60*
_output_shapes
: *!
_class
loc:@v/dense_5/bias*
use_locking(*
validate_shape(*
T0
�
save_5/Assign_61Assignv/dense_5/kernelsave_5/RestoreV2:61*
T0*
use_locking(*
validate_shape(*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel
�
save_5/Assign_62Assignv/dense_5/kernel/Adamsave_5/RestoreV2:62*
use_locking(*#
_class
loc:@v/dense_5/kernel*
validate_shape(*
T0*
_output_shapes

:@ 
�
save_5/Assign_63Assignv/dense_5/kernel/Adam_1save_5/RestoreV2:63*
validate_shape(*
T0*#
_class
loc:@v/dense_5/kernel*
_output_shapes

:@ *
use_locking(
�
save_5/Assign_64Assignv/dense_6/biassave_5/RestoreV2:64*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_6/bias*
T0*
_output_shapes
:
�
save_5/Assign_65Assignv/dense_6/bias/Adamsave_5/RestoreV2:65*
validate_shape(*
T0*!
_class
loc:@v/dense_6/bias*
_output_shapes
:*
use_locking(
�
save_5/Assign_66Assignv/dense_6/bias/Adam_1save_5/RestoreV2:66*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_6/bias
�
save_5/Assign_67Assignv/dense_6/kernelsave_5/RestoreV2:67*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
T0*
use_locking(
�
save_5/Assign_68Assignv/dense_6/kernel/Adamsave_5/RestoreV2:68*
use_locking(*
T0*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel
�
save_5/Assign_69Assignv/dense_6/kernel/Adam_1save_5/RestoreV2:69*
_output_shapes

: *
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_6/kernel
�
save_5/Assign_70Assignv/dense_7/biassave_5/RestoreV2:70*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_7/bias*
validate_shape(
�
save_5/Assign_71Assignv/dense_7/bias/Adamsave_5/RestoreV2:71*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_7/bias
�
save_5/Assign_72Assignv/dense_7/bias/Adam_1save_5/RestoreV2:72*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_7/bias*
T0*
use_locking(
�
save_5/Assign_73Assignv/dense_7/kernelsave_5/RestoreV2:73*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel
�
save_5/Assign_74Assignv/dense_7/kernel/Adamsave_5/RestoreV2:74*#
_class
loc:@v/dense_7/kernel*
use_locking(*
_output_shapes

:*
T0*
validate_shape(
�
save_5/Assign_75Assignv/dense_7/kernel/Adam_1save_5/RestoreV2:75*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
use_locking(*
T0
�
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_45^save_5/Assign_46^save_5/Assign_47^save_5/Assign_48^save_5/Assign_49^save_5/Assign_5^save_5/Assign_50^save_5/Assign_51^save_5/Assign_52^save_5/Assign_53^save_5/Assign_54^save_5/Assign_55^save_5/Assign_56^save_5/Assign_57^save_5/Assign_58^save_5/Assign_59^save_5/Assign_6^save_5/Assign_60^save_5/Assign_61^save_5/Assign_62^save_5/Assign_63^save_5/Assign_64^save_5/Assign_65^save_5/Assign_66^save_5/Assign_67^save_5/Assign_68^save_5/Assign_69^save_5/Assign_7^save_5/Assign_70^save_5/Assign_71^save_5/Assign_72^save_5/Assign_73^save_5/Assign_74^save_5/Assign_75^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
_output_shapes
: *
shape: *
dtype0
�
save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_265d861bc7724086a5662ba28e19f1b4/part*
dtype0*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_6/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_6/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
�
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
�
save_6/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
save_6/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: *
T0
�
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*

axis *
N*
T0*
_output_shapes
:
�
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
�
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
_output_shapes
: *
T0
�
save_6/RestoreV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0
�
!save_6/RestoreV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
validate_shape(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0*
use_locking(
�
save_6/Assign_1Assignbeta1_power_1save_6/RestoreV2:1*
use_locking(*
validate_shape(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
save_6/Assign_2Assignbeta2_powersave_6/RestoreV2:2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0
�
save_6/Assign_3Assignbeta2_power_1save_6/RestoreV2:3*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(
�
save_6/Assign_4Assignpi/dense/biassave_6/RestoreV2:4*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: 
�
save_6/Assign_5Assignpi/dense/bias/Adamsave_6/RestoreV2:5*
use_locking(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_6/Assign_6Assignpi/dense/bias/Adam_1save_6/RestoreV2:6*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(
�
save_6/Assign_7Assignpi/dense/kernelsave_6/RestoreV2:7*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
T0
�
save_6/Assign_8Assignpi/dense/kernel/Adamsave_6/RestoreV2:8*
T0*
_output_shapes

: *
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(
�
save_6/Assign_9Assignpi/dense/kernel/Adam_1save_6/RestoreV2:9*
validate_shape(*
_output_shapes

: *
T0*"
_class
loc:@pi/dense/kernel*
use_locking(
�
save_6/Assign_10Assignpi/dense_1/biassave_6/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_6/Assign_11Assignpi/dense_1/bias/Adamsave_6/RestoreV2:11*
_output_shapes
:*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
validate_shape(
�
save_6/Assign_12Assignpi/dense_1/bias/Adam_1save_6/RestoreV2:12*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
save_6/Assign_13Assignpi/dense_1/kernelsave_6/RestoreV2:13*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

: 
�
save_6/Assign_14Assignpi/dense_1/kernel/Adamsave_6/RestoreV2:14*
_output_shapes

: *
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel
�
save_6/Assign_15Assignpi/dense_1/kernel/Adam_1save_6/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: *
validate_shape(*
T0*
use_locking(
�
save_6/Assign_16Assignpi/dense_2/biassave_6/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
�
save_6/Assign_17Assignpi/dense_2/bias/Adamsave_6/RestoreV2:17*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(
�
save_6/Assign_18Assignpi/dense_2/bias/Adam_1save_6/RestoreV2:18*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_6/Assign_19Assignpi/dense_2/kernelsave_6/RestoreV2:19*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(*
use_locking(
�
save_6/Assign_20Assignpi/dense_2/kernel/Adamsave_6/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_6/Assign_21Assignpi/dense_2/kernel/Adam_1save_6/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:*
use_locking(*
T0*
validate_shape(
�
save_6/Assign_22Assignpi/dense_3/biassave_6/RestoreV2:22*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
T0*
use_locking(
�
save_6/Assign_23Assignpi/dense_3/bias/Adamsave_6/RestoreV2:23*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_3/bias
�
save_6/Assign_24Assignpi/dense_3/bias/Adam_1save_6/RestoreV2:24*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:
�
save_6/Assign_25Assignpi/dense_3/kernelsave_6/RestoreV2:25*
T0*
use_locking(*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
validate_shape(
�
save_6/Assign_26Assignpi/dense_3/kernel/Adamsave_6/RestoreV2:26*
T0*
_output_shapes

:*
use_locking(*$
_class
loc:@pi/dense_3/kernel*
validate_shape(
�
save_6/Assign_27Assignpi/dense_3/kernel/Adam_1save_6/RestoreV2:27*
_output_shapes

:*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_3/kernel
�
save_6/Assign_28Assignv/dense/biassave_6/RestoreV2:28*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
T0*
use_locking(
�
save_6/Assign_29Assignv/dense/bias/Adamsave_6/RestoreV2:29*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
�
save_6/Assign_30Assignv/dense/bias/Adam_1save_6/RestoreV2:30*
use_locking(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
validate_shape(
�
save_6/Assign_31Assignv/dense/kernelsave_6/RestoreV2:31*
_output_shapes

: *!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
T0
�
save_6/Assign_32Assignv/dense/kernel/Adamsave_6/RestoreV2:32*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

: *
T0*
use_locking(
�
save_6/Assign_33Assignv/dense/kernel/Adam_1save_6/RestoreV2:33*
_output_shapes

: *
validate_shape(*!
_class
loc:@v/dense/kernel*
T0*
use_locking(
�
save_6/Assign_34Assignv/dense_1/biassave_6/RestoreV2:34*
use_locking(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
�
save_6/Assign_35Assignv/dense_1/bias/Adamsave_6/RestoreV2:35*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias
�
save_6/Assign_36Assignv/dense_1/bias/Adam_1save_6/RestoreV2:36*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
save_6/Assign_37Assignv/dense_1/kernelsave_6/RestoreV2:37*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
use_locking(
�
save_6/Assign_38Assignv/dense_1/kernel/Adamsave_6/RestoreV2:38*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
T0*
use_locking(*
validate_shape(
�
save_6/Assign_39Assignv/dense_1/kernel/Adam_1save_6/RestoreV2:39*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

: 
�
save_6/Assign_40Assignv/dense_2/biassave_6/RestoreV2:40*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
save_6/Assign_41Assignv/dense_2/bias/Adamsave_6/RestoreV2:41*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0
�
save_6/Assign_42Assignv/dense_2/bias/Adam_1save_6/RestoreV2:42*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
�
save_6/Assign_43Assignv/dense_2/kernelsave_6/RestoreV2:43*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
�
save_6/Assign_44Assignv/dense_2/kernel/Adamsave_6/RestoreV2:44*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:
�
save_6/Assign_45Assignv/dense_2/kernel/Adam_1save_6/RestoreV2:45*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:*
T0*
use_locking(
�
save_6/Assign_46Assignv/dense_3/biassave_6/RestoreV2:46*
validate_shape(*
T0*!
_class
loc:@v/dense_3/bias*
use_locking(*
_output_shapes
:
�
save_6/Assign_47Assignv/dense_3/bias/Adamsave_6/RestoreV2:47*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_3/bias
�
save_6/Assign_48Assignv/dense_3/bias/Adam_1save_6/RestoreV2:48*
T0*
use_locking(*!
_class
loc:@v/dense_3/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_49Assignv/dense_3/kernelsave_6/RestoreV2:49*
_output_shapes

:*
use_locking(*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
T0
�
save_6/Assign_50Assignv/dense_3/kernel/Adamsave_6/RestoreV2:50*
validate_shape(*
T0*
_output_shapes

:*
use_locking(*#
_class
loc:@v/dense_3/kernel
�
save_6/Assign_51Assignv/dense_3/kernel/Adam_1save_6/RestoreV2:51*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
T0*
use_locking(
�
save_6/Assign_52Assignv/dense_4/biassave_6/RestoreV2:52*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
validate_shape(*
use_locking(
�
save_6/Assign_53Assignv/dense_4/bias/Adamsave_6/RestoreV2:53*
validate_shape(*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
T0
�
save_6/Assign_54Assignv/dense_4/bias/Adam_1save_6/RestoreV2:54*
use_locking(*
T0*!
_class
loc:@v/dense_4/bias*
validate_shape(*
_output_shapes
:@
�
save_6/Assign_55Assignv/dense_4/kernelsave_6/RestoreV2:55*#
_class
loc:@v/dense_4/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	�@*
T0
�
save_6/Assign_56Assignv/dense_4/kernel/Adamsave_6/RestoreV2:56*#
_class
loc:@v/dense_4/kernel*
T0*
_output_shapes
:	�@*
validate_shape(*
use_locking(
�
save_6/Assign_57Assignv/dense_4/kernel/Adam_1save_6/RestoreV2:57*#
_class
loc:@v/dense_4/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	�@
�
save_6/Assign_58Assignv/dense_5/biassave_6/RestoreV2:58*
validate_shape(*
T0*
_output_shapes
: *!
_class
loc:@v/dense_5/bias*
use_locking(
�
save_6/Assign_59Assignv/dense_5/bias/Adamsave_6/RestoreV2:59*
_output_shapes
: *
validate_shape(*
T0*!
_class
loc:@v/dense_5/bias*
use_locking(
�
save_6/Assign_60Assignv/dense_5/bias/Adam_1save_6/RestoreV2:60*
T0*
validate_shape(*!
_class
loc:@v/dense_5/bias*
use_locking(*
_output_shapes
: 
�
save_6/Assign_61Assignv/dense_5/kernelsave_6/RestoreV2:61*
validate_shape(*
_output_shapes

:@ *
use_locking(*#
_class
loc:@v/dense_5/kernel*
T0
�
save_6/Assign_62Assignv/dense_5/kernel/Adamsave_6/RestoreV2:62*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_5/kernel*
T0*
_output_shapes

:@ 
�
save_6/Assign_63Assignv/dense_5/kernel/Adam_1save_6/RestoreV2:63*
T0*#
_class
loc:@v/dense_5/kernel*
use_locking(*
_output_shapes

:@ *
validate_shape(
�
save_6/Assign_64Assignv/dense_6/biassave_6/RestoreV2:64*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_6/bias*
_output_shapes
:
�
save_6/Assign_65Assignv/dense_6/bias/Adamsave_6/RestoreV2:65*
T0*!
_class
loc:@v/dense_6/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_6/Assign_66Assignv/dense_6/bias/Adam_1save_6/RestoreV2:66*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_6/bias*
_output_shapes
:
�
save_6/Assign_67Assignv/dense_6/kernelsave_6/RestoreV2:67*
use_locking(*
_output_shapes

: *
validate_shape(*#
_class
loc:@v/dense_6/kernel*
T0
�
save_6/Assign_68Assignv/dense_6/kernel/Adamsave_6/RestoreV2:68*
use_locking(*
T0*
_output_shapes

: *
validate_shape(*#
_class
loc:@v/dense_6/kernel
�
save_6/Assign_69Assignv/dense_6/kernel/Adam_1save_6/RestoreV2:69*
T0*
use_locking(*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
validate_shape(
�
save_6/Assign_70Assignv/dense_7/biassave_6/RestoreV2:70*!
_class
loc:@v/dense_7/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
�
save_6/Assign_71Assignv/dense_7/bias/Adamsave_6/RestoreV2:71*!
_class
loc:@v/dense_7/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
�
save_6/Assign_72Assignv/dense_7/bias/Adam_1save_6/RestoreV2:72*
_output_shapes
:*
T0*!
_class
loc:@v/dense_7/bias*
use_locking(*
validate_shape(
�
save_6/Assign_73Assignv/dense_7/kernelsave_6/RestoreV2:73*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
use_locking(*
T0
�
save_6/Assign_74Assignv/dense_7/kernel/Adamsave_6/RestoreV2:74*
use_locking(*
_output_shapes

:*
validate_shape(*
T0*#
_class
loc:@v/dense_7/kernel
�
save_6/Assign_75Assignv/dense_7/kernel/Adam_1save_6/RestoreV2:75*
_output_shapes

:*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_7/kernel
�
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_45^save_6/Assign_46^save_6/Assign_47^save_6/Assign_48^save_6/Assign_49^save_6/Assign_5^save_6/Assign_50^save_6/Assign_51^save_6/Assign_52^save_6/Assign_53^save_6/Assign_54^save_6/Assign_55^save_6/Assign_56^save_6/Assign_57^save_6/Assign_58^save_6/Assign_59^save_6/Assign_6^save_6/Assign_60^save_6/Assign_61^save_6/Assign_62^save_6/Assign_63^save_6/Assign_64^save_6/Assign_65^save_6/Assign_66^save_6/Assign_67^save_6/Assign_68^save_6/Assign_69^save_6/Assign_7^save_6/Assign_70^save_6/Assign_71^save_6/Assign_72^save_6/Assign_73^save_6/Assign_74^save_6/Assign_75^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
shape: *
_output_shapes
: *
dtype0
�
save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_e3fff1d705fd4c95b0b5dcaf7baeb665/part*
_output_shapes
: *
dtype0
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_7/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_7/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
�
save_7/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
save_7/SaveV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_7/ShardedFilename
�
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*
N*
_output_shapes
:*
T0*

axis 
�
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
�
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
�
save_7/RestoreV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
!save_7/RestoreV2/shape_and_slicesConst*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L
�
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L
�
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes
: 
�
save_7/Assign_1Assignbeta1_power_1save_7/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_7/Assign_2Assignbeta2_powersave_7/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0*
use_locking(
�
save_7/Assign_3Assignbeta2_power_1save_7/RestoreV2:3*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
�
save_7/Assign_4Assignpi/dense/biassave_7/RestoreV2:4*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
use_locking(
�
save_7/Assign_5Assignpi/dense/bias/Adamsave_7/RestoreV2:5* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes
: *
use_locking(
�
save_7/Assign_6Assignpi/dense/bias/Adam_1save_7/RestoreV2:6*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_7/Assign_7Assignpi/dense/kernelsave_7/RestoreV2:7*
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes

: 
�
save_7/Assign_8Assignpi/dense/kernel/Adamsave_7/RestoreV2:8*
validate_shape(*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
T0*
use_locking(
�
save_7/Assign_9Assignpi/dense/kernel/Adam_1save_7/RestoreV2:9*
T0*
use_locking(*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
validate_shape(
�
save_7/Assign_10Assignpi/dense_1/biassave_7/RestoreV2:10*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:
�
save_7/Assign_11Assignpi/dense_1/bias/Adamsave_7/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_7/Assign_12Assignpi/dense_1/bias/Adam_1save_7/RestoreV2:12*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(
�
save_7/Assign_13Assignpi/dense_1/kernelsave_7/RestoreV2:13*
validate_shape(*
_output_shapes

: *
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel
�
save_7/Assign_14Assignpi/dense_1/kernel/Adamsave_7/RestoreV2:14*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

: 
�
save_7/Assign_15Assignpi/dense_1/kernel/Adam_1save_7/RestoreV2:15*
validate_shape(*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0
�
save_7/Assign_16Assignpi/dense_2/biassave_7/RestoreV2:16*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(
�
save_7/Assign_17Assignpi/dense_2/bias/Adamsave_7/RestoreV2:17*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
�
save_7/Assign_18Assignpi/dense_2/bias/Adam_1save_7/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_7/Assign_19Assignpi/dense_2/kernelsave_7/RestoreV2:19*
validate_shape(*
T0*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
use_locking(
�
save_7/Assign_20Assignpi/dense_2/kernel/Adamsave_7/RestoreV2:20*
use_locking(*
validate_shape(*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
T0
�
save_7/Assign_21Assignpi/dense_2/kernel/Adam_1save_7/RestoreV2:21*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
T0*
use_locking(*
validate_shape(
�
save_7/Assign_22Assignpi/dense_3/biassave_7/RestoreV2:22*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_3/bias*
use_locking(*
validate_shape(
�
save_7/Assign_23Assignpi/dense_3/bias/Adamsave_7/RestoreV2:23*
validate_shape(*
T0*"
_class
loc:@pi/dense_3/bias*
use_locking(*
_output_shapes
:
�
save_7/Assign_24Assignpi/dense_3/bias/Adam_1save_7/RestoreV2:24*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
use_locking(*
_output_shapes
:*
T0
�
save_7/Assign_25Assignpi/dense_3/kernelsave_7/RestoreV2:25*
use_locking(*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0
�
save_7/Assign_26Assignpi/dense_3/kernel/Adamsave_7/RestoreV2:26*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*$
_class
loc:@pi/dense_3/kernel
�
save_7/Assign_27Assignpi/dense_3/kernel/Adam_1save_7/RestoreV2:27*
T0*
_output_shapes

:*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
validate_shape(
�
save_7/Assign_28Assignv/dense/biassave_7/RestoreV2:28*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(
�
save_7/Assign_29Assignv/dense/bias/Adamsave_7/RestoreV2:29*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
�
save_7/Assign_30Assignv/dense/bias/Adam_1save_7/RestoreV2:30*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias
�
save_7/Assign_31Assignv/dense/kernelsave_7/RestoreV2:31*
T0*
_output_shapes

: *
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(
�
save_7/Assign_32Assignv/dense/kernel/Adamsave_7/RestoreV2:32*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

: *
use_locking(*
validate_shape(
�
save_7/Assign_33Assignv/dense/kernel/Adam_1save_7/RestoreV2:33*
use_locking(*
T0*
validate_shape(*
_output_shapes

: *!
_class
loc:@v/dense/kernel
�
save_7/Assign_34Assignv/dense_1/biassave_7/RestoreV2:34*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:*
T0
�
save_7/Assign_35Assignv/dense_1/bias/Adamsave_7/RestoreV2:35*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_7/Assign_36Assignv/dense_1/bias/Adam_1save_7/RestoreV2:36*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:
�
save_7/Assign_37Assignv/dense_1/kernelsave_7/RestoreV2:37*
_output_shapes

: *
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_7/Assign_38Assignv/dense_1/kernel/Adamsave_7/RestoreV2:38*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
�
save_7/Assign_39Assignv/dense_1/kernel/Adam_1save_7/RestoreV2:39*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

: 
�
save_7/Assign_40Assignv/dense_2/biassave_7/RestoreV2:40*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
save_7/Assign_41Assignv/dense_2/bias/Adamsave_7/RestoreV2:41*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(*
validate_shape(
�
save_7/Assign_42Assignv/dense_2/bias/Adam_1save_7/RestoreV2:42*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_7/Assign_43Assignv/dense_2/kernelsave_7/RestoreV2:43*
T0*
_output_shapes

:*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_7/Assign_44Assignv/dense_2/kernel/Adamsave_7/RestoreV2:44*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_7/Assign_45Assignv/dense_2/kernel/Adam_1save_7/RestoreV2:45*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
validate_shape(*
T0*
use_locking(
�
save_7/Assign_46Assignv/dense_3/biassave_7/RestoreV2:46*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_3/bias
�
save_7/Assign_47Assignv/dense_3/bias/Adamsave_7/RestoreV2:47*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
T0
�
save_7/Assign_48Assignv/dense_3/bias/Adam_1save_7/RestoreV2:48*
T0*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
use_locking(*
validate_shape(
�
save_7/Assign_49Assignv/dense_3/kernelsave_7/RestoreV2:49*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
T0*
use_locking(
�
save_7/Assign_50Assignv/dense_3/kernel/Adamsave_7/RestoreV2:50*
_output_shapes

:*
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_3/kernel
�
save_7/Assign_51Assignv/dense_3/kernel/Adam_1save_7/RestoreV2:51*
validate_shape(*
_output_shapes

:*
use_locking(*#
_class
loc:@v/dense_3/kernel*
T0
�
save_7/Assign_52Assignv/dense_4/biassave_7/RestoreV2:52*
T0*
validate_shape(*!
_class
loc:@v/dense_4/bias*
use_locking(*
_output_shapes
:@
�
save_7/Assign_53Assignv/dense_4/bias/Adamsave_7/RestoreV2:53*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_4/bias*
_output_shapes
:@
�
save_7/Assign_54Assignv/dense_4/bias/Adam_1save_7/RestoreV2:54*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
validate_shape(*
T0*
use_locking(
�
save_7/Assign_55Assignv/dense_4/kernelsave_7/RestoreV2:55*
T0*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
validate_shape(*
use_locking(
�
save_7/Assign_56Assignv/dense_4/kernel/Adamsave_7/RestoreV2:56*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
T0*
_output_shapes
:	�@
�
save_7/Assign_57Assignv/dense_4/kernel/Adam_1save_7/RestoreV2:57*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
T0*
_output_shapes
:	�@*
use_locking(
�
save_7/Assign_58Assignv/dense_5/biassave_7/RestoreV2:58*!
_class
loc:@v/dense_5/bias*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
�
save_7/Assign_59Assignv/dense_5/bias/Adamsave_7/RestoreV2:59*
use_locking(*
_output_shapes
: *
validate_shape(*!
_class
loc:@v/dense_5/bias*
T0
�
save_7/Assign_60Assignv/dense_5/bias/Adam_1save_7/RestoreV2:60*
use_locking(*
_output_shapes
: *
T0*!
_class
loc:@v/dense_5/bias*
validate_shape(
�
save_7/Assign_61Assignv/dense_5/kernelsave_7/RestoreV2:61*
T0*
validate_shape(*
_output_shapes

:@ *
use_locking(*#
_class
loc:@v/dense_5/kernel
�
save_7/Assign_62Assignv/dense_5/kernel/Adamsave_7/RestoreV2:62*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_5/kernel*
_output_shapes

:@ 
�
save_7/Assign_63Assignv/dense_5/kernel/Adam_1save_7/RestoreV2:63*
use_locking(*
T0*
validate_shape(*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel
�
save_7/Assign_64Assignv/dense_6/biassave_7/RestoreV2:64*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_6/bias
�
save_7/Assign_65Assignv/dense_6/bias/Adamsave_7/RestoreV2:65*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_6/bias
�
save_7/Assign_66Assignv/dense_6/bias/Adam_1save_7/RestoreV2:66*!
_class
loc:@v/dense_6/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
�
save_7/Assign_67Assignv/dense_6/kernelsave_7/RestoreV2:67*
use_locking(*#
_class
loc:@v/dense_6/kernel*
validate_shape(*
T0*
_output_shapes

: 
�
save_7/Assign_68Assignv/dense_6/kernel/Adamsave_7/RestoreV2:68*
use_locking(*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
validate_shape(*
T0
�
save_7/Assign_69Assignv/dense_6/kernel/Adam_1save_7/RestoreV2:69*
use_locking(*
validate_shape(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_6/kernel
�
save_7/Assign_70Assignv/dense_7/biassave_7/RestoreV2:70*!
_class
loc:@v/dense_7/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_7/Assign_71Assignv/dense_7/bias/Adamsave_7/RestoreV2:71*!
_class
loc:@v/dense_7/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
save_7/Assign_72Assignv/dense_7/bias/Adam_1save_7/RestoreV2:72*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_7/bias
�
save_7/Assign_73Assignv/dense_7/kernelsave_7/RestoreV2:73*
use_locking(*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
T0*
_output_shapes

:
�
save_7/Assign_74Assignv/dense_7/kernel/Adamsave_7/RestoreV2:74*
T0*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
validate_shape(
�
save_7/Assign_75Assignv/dense_7/kernel/Adam_1save_7/RestoreV2:75*
_output_shapes

:*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_7/kernel
�
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_45^save_7/Assign_46^save_7/Assign_47^save_7/Assign_48^save_7/Assign_49^save_7/Assign_5^save_7/Assign_50^save_7/Assign_51^save_7/Assign_52^save_7/Assign_53^save_7/Assign_54^save_7/Assign_55^save_7/Assign_56^save_7/Assign_57^save_7/Assign_58^save_7/Assign_59^save_7/Assign_6^save_7/Assign_60^save_7/Assign_61^save_7/Assign_62^save_7/Assign_63^save_7/Assign_64^save_7/Assign_65^save_7/Assign_66^save_7/Assign_67^save_7/Assign_68^save_7/Assign_69^save_7/Assign_7^save_7/Assign_70^save_7/Assign_71^save_7/Assign_72^save_7/Assign_73^save_7/Assign_74^save_7/Assign_75^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
_output_shapes
: *
shape: *
dtype0
�
save_8/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_499fd6a3830a4d71b5f842f7b7ed88d2/part*
dtype0
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_8/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_8/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
�
save_8/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
_output_shapes
:L*
dtype0
�
save_8/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_8/ShardedFilename
�
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
N*

axis *
_output_shapes
:*
T0
�
save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
�
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
_output_shapes
: *
T0
�
save_8/RestoreV2/tensor_namesConst*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
_output_shapes
:L
�
!save_8/RestoreV2/shape_and_slicesConst*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L
�
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_8/AssignAssignbeta1_powersave_8/RestoreV2* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_8/Assign_1Assignbeta1_power_1save_8/RestoreV2:1*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
�
save_8/Assign_2Assignbeta2_powersave_8/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
�
save_8/Assign_3Assignbeta2_power_1save_8/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(
�
save_8/Assign_4Assignpi/dense/biassave_8/RestoreV2:4*
_output_shapes
: *
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0
�
save_8/Assign_5Assignpi/dense/bias/Adamsave_8/RestoreV2:5* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
�
save_8/Assign_6Assignpi/dense/bias/Adam_1save_8/RestoreV2:6*
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
save_8/Assign_7Assignpi/dense/kernelsave_8/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(
�
save_8/Assign_8Assignpi/dense/kernel/Adamsave_8/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
use_locking(*
_output_shapes

: *
validate_shape(
�
save_8/Assign_9Assignpi/dense/kernel/Adam_1save_8/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

: 
�
save_8/Assign_10Assignpi/dense_1/biassave_8/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_8/Assign_11Assignpi/dense_1/bias/Adamsave_8/RestoreV2:11*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
validate_shape(
�
save_8/Assign_12Assignpi/dense_1/bias/Adam_1save_8/RestoreV2:12*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:*
T0
�
save_8/Assign_13Assignpi/dense_1/kernelsave_8/RestoreV2:13*
_output_shapes

: *
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(
�
save_8/Assign_14Assignpi/dense_1/kernel/Adamsave_8/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

: *
validate_shape(*
use_locking(
�
save_8/Assign_15Assignpi/dense_1/kernel/Adam_1save_8/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(
�
save_8/Assign_16Assignpi/dense_2/biassave_8/RestoreV2:16*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(
�
save_8/Assign_17Assignpi/dense_2/bias/Adamsave_8/RestoreV2:17*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
�
save_8/Assign_18Assignpi/dense_2/bias/Adam_1save_8/RestoreV2:18*
_output_shapes
:*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0
�
save_8/Assign_19Assignpi/dense_2/kernelsave_8/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:*
validate_shape(*
use_locking(
�
save_8/Assign_20Assignpi/dense_2/kernel/Adamsave_8/RestoreV2:20*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:
�
save_8/Assign_21Assignpi/dense_2/kernel/Adam_1save_8/RestoreV2:21*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:
�
save_8/Assign_22Assignpi/dense_3/biassave_8/RestoreV2:22*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
T0*
use_locking(
�
save_8/Assign_23Assignpi/dense_3/bias/Adamsave_8/RestoreV2:23*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
use_locking(*
validate_shape(
�
save_8/Assign_24Assignpi/dense_3/bias/Adam_1save_8/RestoreV2:24*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
use_locking(
�
save_8/Assign_25Assignpi/dense_3/kernelsave_8/RestoreV2:25*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:
�
save_8/Assign_26Assignpi/dense_3/kernel/Adamsave_8/RestoreV2:26*
use_locking(*
T0*$
_class
loc:@pi/dense_3/kernel*
validate_shape(*
_output_shapes

:
�
save_8/Assign_27Assignpi/dense_3/kernel/Adam_1save_8/RestoreV2:27*
_output_shapes

:*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
T0*
validate_shape(
�
save_8/Assign_28Assignv/dense/biassave_8/RestoreV2:28*
use_locking(*
_output_shapes
: *
validate_shape(*
T0*
_class
loc:@v/dense/bias
�
save_8/Assign_29Assignv/dense/bias/Adamsave_8/RestoreV2:29*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
T0
�
save_8/Assign_30Assignv/dense/bias/Adam_1save_8/RestoreV2:30*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(
�
save_8/Assign_31Assignv/dense/kernelsave_8/RestoreV2:31*
T0*
validate_shape(*
use_locking(*
_output_shapes

: *!
_class
loc:@v/dense/kernel
�
save_8/Assign_32Assignv/dense/kernel/Adamsave_8/RestoreV2:32*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

: 
�
save_8/Assign_33Assignv/dense/kernel/Adam_1save_8/RestoreV2:33*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

: *
use_locking(*
validate_shape(
�
save_8/Assign_34Assignv/dense_1/biassave_8/RestoreV2:34*
T0*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(
�
save_8/Assign_35Assignv/dense_1/bias/Adamsave_8/RestoreV2:35*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:
�
save_8/Assign_36Assignv/dense_1/bias/Adam_1save_8/RestoreV2:36*
T0*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(
�
save_8/Assign_37Assignv/dense_1/kernelsave_8/RestoreV2:37*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
_output_shapes

: 
�
save_8/Assign_38Assignv/dense_1/kernel/Adamsave_8/RestoreV2:38*
T0*
validate_shape(*
use_locking(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel
�
save_8/Assign_39Assignv/dense_1/kernel/Adam_1save_8/RestoreV2:39*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

: *
T0*
use_locking(
�
save_8/Assign_40Assignv/dense_2/biassave_8/RestoreV2:40*
_output_shapes
:*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_8/Assign_41Assignv/dense_2/bias/Adamsave_8/RestoreV2:41*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0*
validate_shape(
�
save_8/Assign_42Assignv/dense_2/bias/Adam_1save_8/RestoreV2:42*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_8/Assign_43Assignv/dense_2/kernelsave_8/RestoreV2:43*
use_locking(*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:*
validate_shape(
�
save_8/Assign_44Assignv/dense_2/kernel/Adamsave_8/RestoreV2:44*
use_locking(*
_output_shapes

:*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_8/Assign_45Assignv/dense_2/kernel/Adam_1save_8/RestoreV2:45*
use_locking(*
_output_shapes

:*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0
�
save_8/Assign_46Assignv/dense_3/biassave_8/RestoreV2:46*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
use_locking(
�
save_8/Assign_47Assignv/dense_3/bias/Adamsave_8/RestoreV2:47*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_3/bias*
use_locking(
�
save_8/Assign_48Assignv/dense_3/bias/Adam_1save_8/RestoreV2:48*
_output_shapes
:*
T0*
use_locking(*!
_class
loc:@v/dense_3/bias*
validate_shape(
�
save_8/Assign_49Assignv/dense_3/kernelsave_8/RestoreV2:49*
T0*
_output_shapes

:*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
use_locking(
�
save_8/Assign_50Assignv/dense_3/kernel/Adamsave_8/RestoreV2:50*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
T0
�
save_8/Assign_51Assignv/dense_3/kernel/Adam_1save_8/RestoreV2:51*
validate_shape(*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
T0
�
save_8/Assign_52Assignv/dense_4/biassave_8/RestoreV2:52*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_4/bias*
T0*
validate_shape(
�
save_8/Assign_53Assignv/dense_4/bias/Adamsave_8/RestoreV2:53*
T0*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
validate_shape(
�
save_8/Assign_54Assignv/dense_4/bias/Adam_1save_8/RestoreV2:54*!
_class
loc:@v/dense_4/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:@
�
save_8/Assign_55Assignv/dense_4/kernelsave_8/RestoreV2:55*#
_class
loc:@v/dense_4/kernel*
T0*
use_locking(*
_output_shapes
:	�@*
validate_shape(
�
save_8/Assign_56Assignv/dense_4/kernel/Adamsave_8/RestoreV2:56*
_output_shapes
:	�@*
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_4/kernel
�
save_8/Assign_57Assignv/dense_4/kernel/Adam_1save_8/RestoreV2:57*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
T0*
use_locking(*
_output_shapes
:	�@
�
save_8/Assign_58Assignv/dense_5/biassave_8/RestoreV2:58*
validate_shape(*!
_class
loc:@v/dense_5/bias*
use_locking(*
T0*
_output_shapes
: 
�
save_8/Assign_59Assignv/dense_5/bias/Adamsave_8/RestoreV2:59*
_output_shapes
: *!
_class
loc:@v/dense_5/bias*
use_locking(*
T0*
validate_shape(
�
save_8/Assign_60Assignv/dense_5/bias/Adam_1save_8/RestoreV2:60*
use_locking(*!
_class
loc:@v/dense_5/bias*
validate_shape(*
T0*
_output_shapes
: 
�
save_8/Assign_61Assignv/dense_5/kernelsave_8/RestoreV2:61*#
_class
loc:@v/dense_5/kernel*
_output_shapes

:@ *
use_locking(*
T0*
validate_shape(
�
save_8/Assign_62Assignv/dense_5/kernel/Adamsave_8/RestoreV2:62*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_5/kernel*
_output_shapes

:@ 
�
save_8/Assign_63Assignv/dense_5/kernel/Adam_1save_8/RestoreV2:63*
_output_shapes

:@ *
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_5/kernel
�
save_8/Assign_64Assignv/dense_6/biassave_8/RestoreV2:64*!
_class
loc:@v/dense_6/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_8/Assign_65Assignv/dense_6/bias/Adamsave_8/RestoreV2:65*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_6/bias*
_output_shapes
:
�
save_8/Assign_66Assignv/dense_6/bias/Adam_1save_8/RestoreV2:66*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_6/bias*
T0*
use_locking(
�
save_8/Assign_67Assignv/dense_6/kernelsave_8/RestoreV2:67*#
_class
loc:@v/dense_6/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

: 
�
save_8/Assign_68Assignv/dense_6/kernel/Adamsave_8/RestoreV2:68*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
validate_shape(*
use_locking(*
T0
�
save_8/Assign_69Assignv/dense_6/kernel/Adam_1save_8/RestoreV2:69*#
_class
loc:@v/dense_6/kernel*
validate_shape(*
_output_shapes

: *
T0*
use_locking(
�
save_8/Assign_70Assignv/dense_7/biassave_8/RestoreV2:70*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_7/bias
�
save_8/Assign_71Assignv/dense_7/bias/Adamsave_8/RestoreV2:71*
use_locking(*!
_class
loc:@v/dense_7/bias*
_output_shapes
:*
T0*
validate_shape(
�
save_8/Assign_72Assignv/dense_7/bias/Adam_1save_8/RestoreV2:72*!
_class
loc:@v/dense_7/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_8/Assign_73Assignv/dense_7/kernelsave_8/RestoreV2:73*
_output_shapes

:*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_7/kernel
�
save_8/Assign_74Assignv/dense_7/kernel/Adamsave_8/RestoreV2:74*
use_locking(*
validate_shape(*
_output_shapes

:*
T0*#
_class
loc:@v/dense_7/kernel
�
save_8/Assign_75Assignv/dense_7/kernel/Adam_1save_8/RestoreV2:75*
use_locking(*
validate_shape(*
_output_shapes

:*
T0*#
_class
loc:@v/dense_7/kernel
�
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_43^save_8/Assign_44^save_8/Assign_45^save_8/Assign_46^save_8/Assign_47^save_8/Assign_48^save_8/Assign_49^save_8/Assign_5^save_8/Assign_50^save_8/Assign_51^save_8/Assign_52^save_8/Assign_53^save_8/Assign_54^save_8/Assign_55^save_8/Assign_56^save_8/Assign_57^save_8/Assign_58^save_8/Assign_59^save_8/Assign_6^save_8/Assign_60^save_8/Assign_61^save_8/Assign_62^save_8/Assign_63^save_8/Assign_64^save_8/Assign_65^save_8/Assign_66^save_8/Assign_67^save_8/Assign_68^save_8/Assign_69^save_8/Assign_7^save_8/Assign_70^save_8/Assign_71^save_8/Assign_72^save_8/Assign_73^save_8/Assign_74^save_8/Assign_75^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_9/StringJoin/inputs_1Const*<
value3B1 B+_temp_ca6e483b2b334da583bbee4b53dac23e/part*
dtype0*
_output_shapes
: 
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_9/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
�
save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
�
save_9/SaveV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0
�
save_9/SaveV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_9/ShardedFilename
�
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
_output_shapes
:*

axis *
T0*
N
�
save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(
�
save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
�
save_9/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
!save_9/RestoreV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
save_9/Assign_1Assignbeta1_power_1save_9/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_9/Assign_2Assignbeta2_powersave_9/RestoreV2:2*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: *
use_locking(
�
save_9/Assign_3Assignbeta2_power_1save_9/RestoreV2:3*
_output_shapes
: *
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_9/Assign_4Assignpi/dense/biassave_9/RestoreV2:4* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
�
save_9/Assign_5Assignpi/dense/bias/Adamsave_9/RestoreV2:5*
use_locking(*
validate_shape(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0
�
save_9/Assign_6Assignpi/dense/bias/Adam_1save_9/RestoreV2:6*
use_locking(*
validate_shape(*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
�
save_9/Assign_7Assignpi/dense/kernelsave_9/RestoreV2:7*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
T0
�
save_9/Assign_8Assignpi/dense/kernel/Adamsave_9/RestoreV2:8*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(
�
save_9/Assign_9Assignpi/dense/kernel/Adam_1save_9/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

: 
�
save_9/Assign_10Assignpi/dense_1/biassave_9/RestoreV2:10*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias
�
save_9/Assign_11Assignpi/dense_1/bias/Adamsave_9/RestoreV2:11*
T0*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
_output_shapes
:
�
save_9/Assign_12Assignpi/dense_1/bias/Adam_1save_9/RestoreV2:12*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
use_locking(*
T0
�
save_9/Assign_13Assignpi/dense_1/kernelsave_9/RestoreV2:13*
_output_shapes

: *
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
�
save_9/Assign_14Assignpi/dense_1/kernel/Adamsave_9/RestoreV2:14*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: 
�
save_9/Assign_15Assignpi/dense_1/kernel/Adam_1save_9/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
T0*
_output_shapes

: *
validate_shape(
�
save_9/Assign_16Assignpi/dense_2/biassave_9/RestoreV2:16*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
�
save_9/Assign_17Assignpi/dense_2/bias/Adamsave_9/RestoreV2:17*
validate_shape(*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
_output_shapes
:
�
save_9/Assign_18Assignpi/dense_2/bias/Adam_1save_9/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
�
save_9/Assign_19Assignpi/dense_2/kernelsave_9/RestoreV2:19*
T0*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(
�
save_9/Assign_20Assignpi/dense_2/kernel/Adamsave_9/RestoreV2:20*
T0*
_output_shapes

:*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(
�
save_9/Assign_21Assignpi/dense_2/kernel/Adam_1save_9/RestoreV2:21*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
T0
�
save_9/Assign_22Assignpi/dense_3/biassave_9/RestoreV2:22*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias
�
save_9/Assign_23Assignpi/dense_3/bias/Adamsave_9/RestoreV2:23*
validate_shape(*
T0*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:*
use_locking(
�
save_9/Assign_24Assignpi/dense_3/bias/Adam_1save_9/RestoreV2:24*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
T0*
use_locking(
�
save_9/Assign_25Assignpi/dense_3/kernelsave_9/RestoreV2:25*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
T0*
_output_shapes

:*
validate_shape(
�
save_9/Assign_26Assignpi/dense_3/kernel/Adamsave_9/RestoreV2:26*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_3/kernel*
T0*
_output_shapes

:
�
save_9/Assign_27Assignpi/dense_3/kernel/Adam_1save_9/RestoreV2:27*$
_class
loc:@pi/dense_3/kernel*
validate_shape(*
use_locking(*
_output_shapes

:*
T0
�
save_9/Assign_28Assignv/dense/biassave_9/RestoreV2:28*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
�
save_9/Assign_29Assignv/dense/bias/Adamsave_9/RestoreV2:29*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
�
save_9/Assign_30Assignv/dense/bias/Adam_1save_9/RestoreV2:30*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
�
save_9/Assign_31Assignv/dense/kernelsave_9/RestoreV2:31*
_output_shapes

: *!
_class
loc:@v/dense/kernel*
use_locking(*
T0*
validate_shape(
�
save_9/Assign_32Assignv/dense/kernel/Adamsave_9/RestoreV2:32*
_output_shapes

: *
validate_shape(*!
_class
loc:@v/dense/kernel*
T0*
use_locking(
�
save_9/Assign_33Assignv/dense/kernel/Adam_1save_9/RestoreV2:33*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes

: 
�
save_9/Assign_34Assignv/dense_1/biassave_9/RestoreV2:34*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_9/Assign_35Assignv/dense_1/bias/Adamsave_9/RestoreV2:35*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_9/Assign_36Assignv/dense_1/bias/Adam_1save_9/RestoreV2:36*
_output_shapes
:*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(
�
save_9/Assign_37Assignv/dense_1/kernelsave_9/RestoreV2:37*
T0*
_output_shapes

: *
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_9/Assign_38Assignv/dense_1/kernel/Adamsave_9/RestoreV2:38*
_output_shapes

: *
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
�
save_9/Assign_39Assignv/dense_1/kernel/Adam_1save_9/RestoreV2:39*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(*
_output_shapes

: 
�
save_9/Assign_40Assignv/dense_2/biassave_9/RestoreV2:40*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(*
validate_shape(
�
save_9/Assign_41Assignv/dense_2/bias/Adamsave_9/RestoreV2:41*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_9/Assign_42Assignv/dense_2/bias/Adam_1save_9/RestoreV2:42*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_9/Assign_43Assignv/dense_2/kernelsave_9/RestoreV2:43*
_output_shapes

:*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0
�
save_9/Assign_44Assignv/dense_2/kernel/Adamsave_9/RestoreV2:44*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
save_9/Assign_45Assignv/dense_2/kernel/Adam_1save_9/RestoreV2:45*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
use_locking(
�
save_9/Assign_46Assignv/dense_3/biassave_9/RestoreV2:46*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias
�
save_9/Assign_47Assignv/dense_3/bias/Adamsave_9/RestoreV2:47*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_3/bias
�
save_9/Assign_48Assignv/dense_3/bias/Adam_1save_9/RestoreV2:48*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
validate_shape(*
use_locking(*
T0
�
save_9/Assign_49Assignv/dense_3/kernelsave_9/RestoreV2:49*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
use_locking(*
_output_shapes

:*
T0
�
save_9/Assign_50Assignv/dense_3/kernel/Adamsave_9/RestoreV2:50*
T0*
use_locking(*
_output_shapes

:*
validate_shape(*#
_class
loc:@v/dense_3/kernel
�
save_9/Assign_51Assignv/dense_3/kernel/Adam_1save_9/RestoreV2:51*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
use_locking(*
_output_shapes

:*
T0
�
save_9/Assign_52Assignv/dense_4/biassave_9/RestoreV2:52*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
use_locking(*
T0*
validate_shape(
�
save_9/Assign_53Assignv/dense_4/bias/Adamsave_9/RestoreV2:53*!
_class
loc:@v/dense_4/bias*
T0*
_output_shapes
:@*
validate_shape(*
use_locking(
�
save_9/Assign_54Assignv/dense_4/bias/Adam_1save_9/RestoreV2:54*
_output_shapes
:@*
T0*
validate_shape(*!
_class
loc:@v/dense_4/bias*
use_locking(
�
save_9/Assign_55Assignv/dense_4/kernelsave_9/RestoreV2:55*#
_class
loc:@v/dense_4/kernel*
use_locking(*
_output_shapes
:	�@*
validate_shape(*
T0
�
save_9/Assign_56Assignv/dense_4/kernel/Adamsave_9/RestoreV2:56*#
_class
loc:@v/dense_4/kernel*
T0*
validate_shape(*
_output_shapes
:	�@*
use_locking(
�
save_9/Assign_57Assignv/dense_4/kernel/Adam_1save_9/RestoreV2:57*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
T0*
_output_shapes
:	�@
�
save_9/Assign_58Assignv/dense_5/biassave_9/RestoreV2:58*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
T0
�
save_9/Assign_59Assignv/dense_5/bias/Adamsave_9/RestoreV2:59*
validate_shape(*
T0*!
_class
loc:@v/dense_5/bias*
use_locking(*
_output_shapes
: 
�
save_9/Assign_60Assignv/dense_5/bias/Adam_1save_9/RestoreV2:60*
validate_shape(*
T0*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
use_locking(
�
save_9/Assign_61Assignv/dense_5/kernelsave_9/RestoreV2:61*
validate_shape(*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel*
T0*
use_locking(
�
save_9/Assign_62Assignv/dense_5/kernel/Adamsave_9/RestoreV2:62*
validate_shape(*
T0*
_output_shapes

:@ *
use_locking(*#
_class
loc:@v/dense_5/kernel
�
save_9/Assign_63Assignv/dense_5/kernel/Adam_1save_9/RestoreV2:63*
use_locking(*#
_class
loc:@v/dense_5/kernel*
T0*
validate_shape(*
_output_shapes

:@ 
�
save_9/Assign_64Assignv/dense_6/biassave_9/RestoreV2:64*!
_class
loc:@v/dense_6/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
�
save_9/Assign_65Assignv/dense_6/bias/Adamsave_9/RestoreV2:65*!
_class
loc:@v/dense_6/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
�
save_9/Assign_66Assignv/dense_6/bias/Adam_1save_9/RestoreV2:66*
T0*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
validate_shape(*
use_locking(
�
save_9/Assign_67Assignv/dense_6/kernelsave_9/RestoreV2:67*
T0*
use_locking(*#
_class
loc:@v/dense_6/kernel*
validate_shape(*
_output_shapes

: 
�
save_9/Assign_68Assignv/dense_6/kernel/Adamsave_9/RestoreV2:68*#
_class
loc:@v/dense_6/kernel*
use_locking(*
T0*
_output_shapes

: *
validate_shape(
�
save_9/Assign_69Assignv/dense_6/kernel/Adam_1save_9/RestoreV2:69*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*#
_class
loc:@v/dense_6/kernel
�
save_9/Assign_70Assignv/dense_7/biassave_9/RestoreV2:70*
use_locking(*!
_class
loc:@v/dense_7/bias*
validate_shape(*
_output_shapes
:*
T0
�
save_9/Assign_71Assignv/dense_7/bias/Adamsave_9/RestoreV2:71*!
_class
loc:@v/dense_7/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_9/Assign_72Assignv/dense_7/bias/Adam_1save_9/RestoreV2:72*
T0*!
_class
loc:@v/dense_7/bias*
use_locking(*
validate_shape(*
_output_shapes
:
�
save_9/Assign_73Assignv/dense_7/kernelsave_9/RestoreV2:73*
validate_shape(*#
_class
loc:@v/dense_7/kernel*
_output_shapes

:*
T0*
use_locking(
�
save_9/Assign_74Assignv/dense_7/kernel/Adamsave_9/RestoreV2:74*
validate_shape(*#
_class
loc:@v/dense_7/kernel*
T0*
_output_shapes

:*
use_locking(
�
save_9/Assign_75Assignv/dense_7/kernel/Adam_1save_9/RestoreV2:75*
use_locking(*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
T0
�
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_43^save_9/Assign_44^save_9/Assign_45^save_9/Assign_46^save_9/Assign_47^save_9/Assign_48^save_9/Assign_49^save_9/Assign_5^save_9/Assign_50^save_9/Assign_51^save_9/Assign_52^save_9/Assign_53^save_9/Assign_54^save_9/Assign_55^save_9/Assign_56^save_9/Assign_57^save_9/Assign_58^save_9/Assign_59^save_9/Assign_6^save_9/Assign_60^save_9/Assign_61^save_9/Assign_62^save_9/Assign_63^save_9/Assign_64^save_9/Assign_65^save_9/Assign_66^save_9/Assign_67^save_9/Assign_68^save_9/Assign_69^save_9/Assign_7^save_9/Assign_70^save_9/Assign_71^save_9/Assign_72^save_9/Assign_73^save_9/Assign_74^save_9/Assign_75^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
_output_shapes
: *
shape: *
dtype0
�
save_10/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_e6d2bc42ae5c4ef197d6d74cb3d7403f/part*
dtype0
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_10/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
�
save_10/SaveV2/tensor_namesConst*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
_output_shapes
:L
�
save_10/SaveV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2**
_class 
loc:@save_10/ShardedFilename*
_output_shapes
: *
T0
�
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
N*
_output_shapes
:*

axis *
T0
�
save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(
�
save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
T0*
_output_shapes
: 
�
save_10/RestoreV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
"save_10/RestoreV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(
�
save_10/Assign_1Assignbeta1_power_1save_10/RestoreV2:1*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0*
use_locking(
�
save_10/Assign_2Assignbeta2_powersave_10/RestoreV2:2*
_output_shapes
: *
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias*
T0
�
save_10/Assign_3Assignbeta2_power_1save_10/RestoreV2:3*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
�
save_10/Assign_4Assignpi/dense/biassave_10/RestoreV2:4* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
�
save_10/Assign_5Assignpi/dense/bias/Adamsave_10/RestoreV2:5*
use_locking(* 
_class
loc:@pi/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
�
save_10/Assign_6Assignpi/dense/bias/Adam_1save_10/RestoreV2:6*
use_locking(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0*
validate_shape(
�
save_10/Assign_7Assignpi/dense/kernelsave_10/RestoreV2:7*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
T0*
validate_shape(
�
save_10/Assign_8Assignpi/dense/kernel/Adamsave_10/RestoreV2:8*
validate_shape(*
_output_shapes

: *
use_locking(*"
_class
loc:@pi/dense/kernel*
T0
�
save_10/Assign_9Assignpi/dense/kernel/Adam_1save_10/RestoreV2:9*
validate_shape(*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
use_locking(
�
save_10/Assign_10Assignpi/dense_1/biassave_10/RestoreV2:10*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
T0
�
save_10/Assign_11Assignpi/dense_1/bias/Adamsave_10/RestoreV2:11*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias
�
save_10/Assign_12Assignpi/dense_1/bias/Adam_1save_10/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_10/Assign_13Assignpi/dense_1/kernelsave_10/RestoreV2:13*
validate_shape(*
use_locking(*
_output_shapes

: *
T0*$
_class
loc:@pi/dense_1/kernel
�
save_10/Assign_14Assignpi/dense_1/kernel/Adamsave_10/RestoreV2:14*
_output_shapes

: *
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel
�
save_10/Assign_15Assignpi/dense_1/kernel/Adam_1save_10/RestoreV2:15*
use_locking(*
T0*
validate_shape(*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel
�
save_10/Assign_16Assignpi/dense_2/biassave_10/RestoreV2:16*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
�
save_10/Assign_17Assignpi/dense_2/bias/Adamsave_10/RestoreV2:17*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
�
save_10/Assign_18Assignpi/dense_2/bias/Adam_1save_10/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
�
save_10/Assign_19Assignpi/dense_2/kernelsave_10/RestoreV2:19*
use_locking(*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(
�
save_10/Assign_20Assignpi/dense_2/kernel/Adamsave_10/RestoreV2:20*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
T0
�
save_10/Assign_21Assignpi/dense_2/kernel/Adam_1save_10/RestoreV2:21*
T0*
validate_shape(*
_output_shapes

:*
use_locking(*$
_class
loc:@pi/dense_2/kernel
�
save_10/Assign_22Assignpi/dense_3/biassave_10/RestoreV2:22*
validate_shape(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
use_locking(
�
save_10/Assign_23Assignpi/dense_3/bias/Adamsave_10/RestoreV2:23*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
T0
�
save_10/Assign_24Assignpi/dense_3/bias/Adam_1save_10/RestoreV2:24*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*"
_class
loc:@pi/dense_3/bias
�
save_10/Assign_25Assignpi/dense_3/kernelsave_10/RestoreV2:25*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
validate_shape(*
use_locking(*
T0
�
save_10/Assign_26Assignpi/dense_3/kernel/Adamsave_10/RestoreV2:26*$
_class
loc:@pi/dense_3/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_10/Assign_27Assignpi/dense_3/kernel/Adam_1save_10/RestoreV2:27*
_output_shapes

:*
use_locking(*$
_class
loc:@pi/dense_3/kernel*
T0*
validate_shape(
�
save_10/Assign_28Assignv/dense/biassave_10/RestoreV2:28*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
�
save_10/Assign_29Assignv/dense/bias/Adamsave_10/RestoreV2:29*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
�
save_10/Assign_30Assignv/dense/bias/Adam_1save_10/RestoreV2:30*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_10/Assign_31Assignv/dense/kernelsave_10/RestoreV2:31*
_output_shapes

: *
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_10/Assign_32Assignv/dense/kernel/Adamsave_10/RestoreV2:32*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

: 
�
save_10/Assign_33Assignv/dense/kernel/Adam_1save_10/RestoreV2:33*!
_class
loc:@v/dense/kernel*
_output_shapes

: *
T0*
use_locking(*
validate_shape(
�
save_10/Assign_34Assignv/dense_1/biassave_10/RestoreV2:34*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_10/Assign_35Assignv/dense_1/bias/Adamsave_10/RestoreV2:35*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0
�
save_10/Assign_36Assignv/dense_1/bias/Adam_1save_10/RestoreV2:36*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:
�
save_10/Assign_37Assignv/dense_1/kernelsave_10/RestoreV2:37*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_10/Assign_38Assignv/dense_1/kernel/Adamsave_10/RestoreV2:38*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
_output_shapes

: 
�
save_10/Assign_39Assignv/dense_1/kernel/Adam_1save_10/RestoreV2:39*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: 
�
save_10/Assign_40Assignv/dense_2/biassave_10/RestoreV2:40*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_10/Assign_41Assignv/dense_2/bias/Adamsave_10/RestoreV2:41*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(
�
save_10/Assign_42Assignv/dense_2/bias/Adam_1save_10/RestoreV2:42*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
�
save_10/Assign_43Assignv/dense_2/kernelsave_10/RestoreV2:43*
_output_shapes

:*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_2/kernel
�
save_10/Assign_44Assignv/dense_2/kernel/Adamsave_10/RestoreV2:44*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
validate_shape(
�
save_10/Assign_45Assignv/dense_2/kernel/Adam_1save_10/RestoreV2:45*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
validate_shape(*
T0
�
save_10/Assign_46Assignv/dense_3/biassave_10/RestoreV2:46*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_3/bias
�
save_10/Assign_47Assignv/dense_3/bias/Adamsave_10/RestoreV2:47*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_3/bias*
T0*
_output_shapes
:
�
save_10/Assign_48Assignv/dense_3/bias/Adam_1save_10/RestoreV2:48*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_3/bias*
T0*
validate_shape(
�
save_10/Assign_49Assignv/dense_3/kernelsave_10/RestoreV2:49*
T0*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
validate_shape(
�
save_10/Assign_50Assignv/dense_3/kernel/Adamsave_10/RestoreV2:50*#
_class
loc:@v/dense_3/kernel*
T0*
use_locking(*
_output_shapes

:*
validate_shape(
�
save_10/Assign_51Assignv/dense_3/kernel/Adam_1save_10/RestoreV2:51*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@v/dense_3/kernel*
validate_shape(
�
save_10/Assign_52Assignv/dense_4/biassave_10/RestoreV2:52*
T0*
validate_shape(*!
_class
loc:@v/dense_4/bias*
use_locking(*
_output_shapes
:@
�
save_10/Assign_53Assignv/dense_4/bias/Adamsave_10/RestoreV2:53*
T0*
validate_shape(*!
_class
loc:@v/dense_4/bias*
_output_shapes
:@*
use_locking(
�
save_10/Assign_54Assignv/dense_4/bias/Adam_1save_10/RestoreV2:54*
T0*!
_class
loc:@v/dense_4/bias*
validate_shape(*
use_locking(*
_output_shapes
:@
�
save_10/Assign_55Assignv/dense_4/kernelsave_10/RestoreV2:55*
use_locking(*
T0*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
validate_shape(
�
save_10/Assign_56Assignv/dense_4/kernel/Adamsave_10/RestoreV2:56*
_output_shapes
:	�@*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
T0
�
save_10/Assign_57Assignv/dense_4/kernel/Adam_1save_10/RestoreV2:57*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel
�
save_10/Assign_58Assignv/dense_5/biassave_10/RestoreV2:58*
T0*
_output_shapes
: *
use_locking(*!
_class
loc:@v/dense_5/bias*
validate_shape(
�
save_10/Assign_59Assignv/dense_5/bias/Adamsave_10/RestoreV2:59*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*!
_class
loc:@v/dense_5/bias
�
save_10/Assign_60Assignv/dense_5/bias/Adam_1save_10/RestoreV2:60*!
_class
loc:@v/dense_5/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
�
save_10/Assign_61Assignv/dense_5/kernelsave_10/RestoreV2:61*
_output_shapes

:@ *
T0*
use_locking(*#
_class
loc:@v/dense_5/kernel*
validate_shape(
�
save_10/Assign_62Assignv/dense_5/kernel/Adamsave_10/RestoreV2:62*#
_class
loc:@v/dense_5/kernel*
validate_shape(*
_output_shapes

:@ *
T0*
use_locking(
�
save_10/Assign_63Assignv/dense_5/kernel/Adam_1save_10/RestoreV2:63*
validate_shape(*#
_class
loc:@v/dense_5/kernel*
use_locking(*
_output_shapes

:@ *
T0
�
save_10/Assign_64Assignv/dense_6/biassave_10/RestoreV2:64*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_6/bias
�
save_10/Assign_65Assignv/dense_6/bias/Adamsave_10/RestoreV2:65*
use_locking(*!
_class
loc:@v/dense_6/bias*
T0*
_output_shapes
:*
validate_shape(
�
save_10/Assign_66Assignv/dense_6/bias/Adam_1save_10/RestoreV2:66*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
T0*
use_locking(*
validate_shape(
�
save_10/Assign_67Assignv/dense_6/kernelsave_10/RestoreV2:67*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
validate_shape(*
use_locking(*
T0
�
save_10/Assign_68Assignv/dense_6/kernel/Adamsave_10/RestoreV2:68*
validate_shape(*
use_locking(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_6/kernel
�
save_10/Assign_69Assignv/dense_6/kernel/Adam_1save_10/RestoreV2:69*
T0*
_output_shapes

: *
use_locking(*#
_class
loc:@v/dense_6/kernel*
validate_shape(
�
save_10/Assign_70Assignv/dense_7/biassave_10/RestoreV2:70*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_7/bias*
use_locking(
�
save_10/Assign_71Assignv/dense_7/bias/Adamsave_10/RestoreV2:71*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_7/bias*
use_locking(
�
save_10/Assign_72Assignv/dense_7/bias/Adam_1save_10/RestoreV2:72*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_7/bias
�
save_10/Assign_73Assignv/dense_7/kernelsave_10/RestoreV2:73*
T0*
_output_shapes

:*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_7/kernel
�
save_10/Assign_74Assignv/dense_7/kernel/Adamsave_10/RestoreV2:74*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
use_locking(*
T0
�
save_10/Assign_75Assignv/dense_7/kernel/Adam_1save_10/RestoreV2:75*
validate_shape(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
use_locking(
�
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_40^save_10/Assign_41^save_10/Assign_42^save_10/Assign_43^save_10/Assign_44^save_10/Assign_45^save_10/Assign_46^save_10/Assign_47^save_10/Assign_48^save_10/Assign_49^save_10/Assign_5^save_10/Assign_50^save_10/Assign_51^save_10/Assign_52^save_10/Assign_53^save_10/Assign_54^save_10/Assign_55^save_10/Assign_56^save_10/Assign_57^save_10/Assign_58^save_10/Assign_59^save_10/Assign_6^save_10/Assign_60^save_10/Assign_61^save_10/Assign_62^save_10/Assign_63^save_10/Assign_64^save_10/Assign_65^save_10/Assign_66^save_10/Assign_67^save_10/Assign_68^save_10/Assign_69^save_10/Assign_7^save_10/Assign_70^save_10/Assign_71^save_10/Assign_72^save_10/Assign_73^save_10/Assign_74^save_10/Assign_75^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_11/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3bb64f813b0b42b2b91ad144ea607de1/part
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_11/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_11/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
�
save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
�
save_11/SaveV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0
�
save_11/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
T0**
_class 
loc:@save_11/ShardedFilename*
_output_shapes
: 
�
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*
T0*
_output_shapes
:*

axis *
N
�
save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(
�
save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
T0*
_output_shapes
: 
�
save_11/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
"save_11/RestoreV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_11/AssignAssignbeta1_powersave_11/RestoreV2*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(*
validate_shape(
�
save_11/Assign_1Assignbeta1_power_1save_11/RestoreV2:1*
T0*
use_locking(*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias
�
save_11/Assign_2Assignbeta2_powersave_11/RestoreV2:2* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
�
save_11/Assign_3Assignbeta2_power_1save_11/RestoreV2:3*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
�
save_11/Assign_4Assignpi/dense/biassave_11/RestoreV2:4*
use_locking(*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_11/Assign_5Assignpi/dense/bias/Adamsave_11/RestoreV2:5* 
_class
loc:@pi/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
�
save_11/Assign_6Assignpi/dense/bias/Adam_1save_11/RestoreV2:6*
T0*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
�
save_11/Assign_7Assignpi/dense/kernelsave_11/RestoreV2:7*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
validate_shape(
�
save_11/Assign_8Assignpi/dense/kernel/Adamsave_11/RestoreV2:8*
validate_shape(*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
_output_shapes

: 
�
save_11/Assign_9Assignpi/dense/kernel/Adam_1save_11/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
use_locking(*
T0*
_output_shapes

: *
validate_shape(
�
save_11/Assign_10Assignpi/dense_1/biassave_11/RestoreV2:10*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
T0
�
save_11/Assign_11Assignpi/dense_1/bias/Adamsave_11/RestoreV2:11*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
�
save_11/Assign_12Assignpi/dense_1/bias/Adam_1save_11/RestoreV2:12*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
validate_shape(
�
save_11/Assign_13Assignpi/dense_1/kernelsave_11/RestoreV2:13*
use_locking(*
_output_shapes

: *
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0
�
save_11/Assign_14Assignpi/dense_1/kernel/Adamsave_11/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

: *
use_locking(
�
save_11/Assign_15Assignpi/dense_1/kernel/Adam_1save_11/RestoreV2:15*
validate_shape(*
_output_shapes

: *
T0*$
_class
loc:@pi/dense_1/kernel*
use_locking(
�
save_11/Assign_16Assignpi/dense_2/biassave_11/RestoreV2:16*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0
�
save_11/Assign_17Assignpi/dense_2/bias/Adamsave_11/RestoreV2:17*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias
�
save_11/Assign_18Assignpi/dense_2/bias/Adam_1save_11/RestoreV2:18*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias
�
save_11/Assign_19Assignpi/dense_2/kernelsave_11/RestoreV2:19*
T0*
_output_shapes

:*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
�
save_11/Assign_20Assignpi/dense_2/kernel/Adamsave_11/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
save_11/Assign_21Assignpi/dense_2/kernel/Adam_1save_11/RestoreV2:21*
T0*
_output_shapes

:*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel
�
save_11/Assign_22Assignpi/dense_3/biassave_11/RestoreV2:22*"
_class
loc:@pi/dense_3/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_11/Assign_23Assignpi/dense_3/bias/Adamsave_11/RestoreV2:23*
_output_shapes
:*
T0*
use_locking(*"
_class
loc:@pi/dense_3/bias*
validate_shape(
�
save_11/Assign_24Assignpi/dense_3/bias/Adam_1save_11/RestoreV2:24*
validate_shape(*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias
�
save_11/Assign_25Assignpi/dense_3/kernelsave_11/RestoreV2:25*
_output_shapes

:*
T0*
use_locking(*$
_class
loc:@pi/dense_3/kernel*
validate_shape(
�
save_11/Assign_26Assignpi/dense_3/kernel/Adamsave_11/RestoreV2:26*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
T0
�
save_11/Assign_27Assignpi/dense_3/kernel/Adam_1save_11/RestoreV2:27*
_output_shapes

:*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_3/kernel
�
save_11/Assign_28Assignv/dense/biassave_11/RestoreV2:28*
use_locking(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
validate_shape(
�
save_11/Assign_29Assignv/dense/bias/Adamsave_11/RestoreV2:29*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
�
save_11/Assign_30Assignv/dense/bias/Adam_1save_11/RestoreV2:30*
_output_shapes
: *
T0*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias
�
save_11/Assign_31Assignv/dense/kernelsave_11/RestoreV2:31*
T0*
validate_shape(*
use_locking(*
_output_shapes

: *!
_class
loc:@v/dense/kernel
�
save_11/Assign_32Assignv/dense/kernel/Adamsave_11/RestoreV2:32*
_output_shapes

: *
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel
�
save_11/Assign_33Assignv/dense/kernel/Adam_1save_11/RestoreV2:33*
_output_shapes

: *
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
�
save_11/Assign_34Assignv/dense_1/biassave_11/RestoreV2:34*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:
�
save_11/Assign_35Assignv/dense_1/bias/Adamsave_11/RestoreV2:35*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(
�
save_11/Assign_36Assignv/dense_1/bias/Adam_1save_11/RestoreV2:36*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
save_11/Assign_37Assignv/dense_1/kernelsave_11/RestoreV2:37*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
�
save_11/Assign_38Assignv/dense_1/kernel/Adamsave_11/RestoreV2:38*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0
�
save_11/Assign_39Assignv/dense_1/kernel/Adam_1save_11/RestoreV2:39*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

: *
T0
�
save_11/Assign_40Assignv/dense_2/biassave_11/RestoreV2:40*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_11/Assign_41Assignv/dense_2/bias/Adamsave_11/RestoreV2:41*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_11/Assign_42Assignv/dense_2/bias/Adam_1save_11/RestoreV2:42*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:
�
save_11/Assign_43Assignv/dense_2/kernelsave_11/RestoreV2:43*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:*
validate_shape(
�
save_11/Assign_44Assignv/dense_2/kernel/Adamsave_11/RestoreV2:44*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
save_11/Assign_45Assignv/dense_2/kernel/Adam_1save_11/RestoreV2:45*
_output_shapes

:*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(
�
save_11/Assign_46Assignv/dense_3/biassave_11/RestoreV2:46*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
T0
�
save_11/Assign_47Assignv/dense_3/bias/Adamsave_11/RestoreV2:47*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_3/bias
�
save_11/Assign_48Assignv/dense_3/bias/Adam_1save_11/RestoreV2:48*
validate_shape(*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
T0*
use_locking(
�
save_11/Assign_49Assignv/dense_3/kernelsave_11/RestoreV2:49*
validate_shape(*
T0*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
use_locking(
�
save_11/Assign_50Assignv/dense_3/kernel/Adamsave_11/RestoreV2:50*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
use_locking(*
T0*
_output_shapes

:
�
save_11/Assign_51Assignv/dense_3/kernel/Adam_1save_11/RestoreV2:51*
_output_shapes

:*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_3/kernel
�
save_11/Assign_52Assignv/dense_4/biassave_11/RestoreV2:52*
_output_shapes
:@*
T0*
validate_shape(*!
_class
loc:@v/dense_4/bias*
use_locking(
�
save_11/Assign_53Assignv/dense_4/bias/Adamsave_11/RestoreV2:53*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_4/bias*
_output_shapes
:@*
T0
�
save_11/Assign_54Assignv/dense_4/bias/Adam_1save_11/RestoreV2:54*
T0*
_output_shapes
:@*
validate_shape(*!
_class
loc:@v/dense_4/bias*
use_locking(
�
save_11/Assign_55Assignv/dense_4/kernelsave_11/RestoreV2:55*
use_locking(*
_output_shapes
:	�@*
T0*#
_class
loc:@v/dense_4/kernel*
validate_shape(
�
save_11/Assign_56Assignv/dense_4/kernel/Adamsave_11/RestoreV2:56*
_output_shapes
:	�@*
T0*#
_class
loc:@v/dense_4/kernel*
use_locking(*
validate_shape(
�
save_11/Assign_57Assignv/dense_4/kernel/Adam_1save_11/RestoreV2:57*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
T0*
_output_shapes
:	�@*
use_locking(
�
save_11/Assign_58Assignv/dense_5/biassave_11/RestoreV2:58*
T0*
use_locking(*!
_class
loc:@v/dense_5/bias*
validate_shape(*
_output_shapes
: 
�
save_11/Assign_59Assignv/dense_5/bias/Adamsave_11/RestoreV2:59*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*!
_class
loc:@v/dense_5/bias
�
save_11/Assign_60Assignv/dense_5/bias/Adam_1save_11/RestoreV2:60*
T0*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
validate_shape(*
use_locking(
�
save_11/Assign_61Assignv/dense_5/kernelsave_11/RestoreV2:61*
T0*
_output_shapes

:@ *
validate_shape(*
use_locking(*#
_class
loc:@v/dense_5/kernel
�
save_11/Assign_62Assignv/dense_5/kernel/Adamsave_11/RestoreV2:62*
validate_shape(*
_output_shapes

:@ *
T0*
use_locking(*#
_class
loc:@v/dense_5/kernel
�
save_11/Assign_63Assignv/dense_5/kernel/Adam_1save_11/RestoreV2:63*
validate_shape(*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel*
use_locking(*
T0
�
save_11/Assign_64Assignv/dense_6/biassave_11/RestoreV2:64*
use_locking(*!
_class
loc:@v/dense_6/bias*
T0*
_output_shapes
:*
validate_shape(
�
save_11/Assign_65Assignv/dense_6/bias/Adamsave_11/RestoreV2:65*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_6/bias*
_output_shapes
:
�
save_11/Assign_66Assignv/dense_6/bias/Adam_1save_11/RestoreV2:66*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_6/bias*
_output_shapes
:
�
save_11/Assign_67Assignv/dense_6/kernelsave_11/RestoreV2:67*
validate_shape(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_6/kernel*
use_locking(
�
save_11/Assign_68Assignv/dense_6/kernel/Adamsave_11/RestoreV2:68*
T0*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
use_locking(
�
save_11/Assign_69Assignv/dense_6/kernel/Adam_1save_11/RestoreV2:69*
use_locking(*
_output_shapes

: *
validate_shape(*
T0*#
_class
loc:@v/dense_6/kernel
�
save_11/Assign_70Assignv/dense_7/biassave_11/RestoreV2:70*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_7/bias
�
save_11/Assign_71Assignv/dense_7/bias/Adamsave_11/RestoreV2:71*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_7/bias*
_output_shapes
:
�
save_11/Assign_72Assignv/dense_7/bias/Adam_1save_11/RestoreV2:72*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_7/bias*
T0
�
save_11/Assign_73Assignv/dense_7/kernelsave_11/RestoreV2:73*#
_class
loc:@v/dense_7/kernel*
T0*
use_locking(*
_output_shapes

:*
validate_shape(
�
save_11/Assign_74Assignv/dense_7/kernel/Adamsave_11/RestoreV2:74*
use_locking(*
T0*
_output_shapes

:*
validate_shape(*#
_class
loc:@v/dense_7/kernel
�
save_11/Assign_75Assignv/dense_7/kernel/Adam_1save_11/RestoreV2:75*
T0*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_20^save_11/Assign_21^save_11/Assign_22^save_11/Assign_23^save_11/Assign_24^save_11/Assign_25^save_11/Assign_26^save_11/Assign_27^save_11/Assign_28^save_11/Assign_29^save_11/Assign_3^save_11/Assign_30^save_11/Assign_31^save_11/Assign_32^save_11/Assign_33^save_11/Assign_34^save_11/Assign_35^save_11/Assign_36^save_11/Assign_37^save_11/Assign_38^save_11/Assign_39^save_11/Assign_4^save_11/Assign_40^save_11/Assign_41^save_11/Assign_42^save_11/Assign_43^save_11/Assign_44^save_11/Assign_45^save_11/Assign_46^save_11/Assign_47^save_11/Assign_48^save_11/Assign_49^save_11/Assign_5^save_11/Assign_50^save_11/Assign_51^save_11/Assign_52^save_11/Assign_53^save_11/Assign_54^save_11/Assign_55^save_11/Assign_56^save_11/Assign_57^save_11/Assign_58^save_11/Assign_59^save_11/Assign_6^save_11/Assign_60^save_11/Assign_61^save_11/Assign_62^save_11/Assign_63^save_11/Assign_64^save_11/Assign_65^save_11/Assign_66^save_11/Assign_67^save_11/Assign_68^save_11/Assign_69^save_11/Assign_7^save_11/Assign_70^save_11/Assign_71^save_11/Assign_72^save_11/Assign_73^save_11/Assign_74^save_11/Assign_75^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard
\
save_12/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
t
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
shape: *
_output_shapes
: *
dtype0
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_12/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_e5a7b1379b13457b846a50a588abaca0/part*
_output_shapes
: 
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
T
save_12/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_12/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
�
save_12/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
_output_shapes
:L*
dtype0
�
save_12/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
T0**
_class 
loc:@save_12/ShardedFilename*
_output_shapes
: 
�
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*

axis *
T0*
_output_shapes
:*
N
�
save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(
�
save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
T0*
_output_shapes
: 
�
save_12/RestoreV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0
�
"save_12/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_12/AssignAssignbeta1_powersave_12/RestoreV2*
validate_shape(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
use_locking(*
T0
�
save_12/Assign_1Assignbeta1_power_1save_12/RestoreV2:1*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
�
save_12/Assign_2Assignbeta2_powersave_12/RestoreV2:2*
T0*
validate_shape(*
use_locking(*
_output_shapes
: * 
_class
loc:@pi/dense/bias
�
save_12/Assign_3Assignbeta2_power_1save_12/RestoreV2:3*
T0*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(
�
save_12/Assign_4Assignpi/dense/biassave_12/RestoreV2:4*
_output_shapes
: *
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_12/Assign_5Assignpi/dense/bias/Adamsave_12/RestoreV2:5*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
�
save_12/Assign_6Assignpi/dense/bias/Adam_1save_12/RestoreV2:6*
_output_shapes
: *
T0*
validate_shape(*
use_locking(* 
_class
loc:@pi/dense/bias
�
save_12/Assign_7Assignpi/dense/kernelsave_12/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes

: *
T0*
use_locking(
�
save_12/Assign_8Assignpi/dense/kernel/Adamsave_12/RestoreV2:8*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(*
use_locking(
�
save_12/Assign_9Assignpi/dense/kernel/Adam_1save_12/RestoreV2:9*
_output_shapes

: *
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
validate_shape(
�
save_12/Assign_10Assignpi/dense_1/biassave_12/RestoreV2:10*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:
�
save_12/Assign_11Assignpi/dense_1/bias/Adamsave_12/RestoreV2:11*
_output_shapes
:*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
use_locking(
�
save_12/Assign_12Assignpi/dense_1/bias/Adam_1save_12/RestoreV2:12*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
_output_shapes
:*
validate_shape(
�
save_12/Assign_13Assignpi/dense_1/kernelsave_12/RestoreV2:13*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

: *
use_locking(
�
save_12/Assign_14Assignpi/dense_1/kernel/Adamsave_12/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

: *
validate_shape(*
use_locking(
�
save_12/Assign_15Assignpi/dense_1/kernel/Adam_1save_12/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

: 
�
save_12/Assign_16Assignpi/dense_2/biassave_12/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias*
T0
�
save_12/Assign_17Assignpi/dense_2/bias/Adamsave_12/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
�
save_12/Assign_18Assignpi/dense_2/bias/Adam_1save_12/RestoreV2:18*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
use_locking(
�
save_12/Assign_19Assignpi/dense_2/kernelsave_12/RestoreV2:19*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:
�
save_12/Assign_20Assignpi/dense_2/kernel/Adamsave_12/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:*
T0
�
save_12/Assign_21Assignpi/dense_2/kernel/Adam_1save_12/RestoreV2:21*
_output_shapes

:*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(
�
save_12/Assign_22Assignpi/dense_3/biassave_12/RestoreV2:22*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_3/bias
�
save_12/Assign_23Assignpi/dense_3/bias/Adamsave_12/RestoreV2:23*
T0*
use_locking(*"
_class
loc:@pi/dense_3/bias*
validate_shape(*
_output_shapes
:
�
save_12/Assign_24Assignpi/dense_3/bias/Adam_1save_12/RestoreV2:24*
T0*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_12/Assign_25Assignpi/dense_3/kernelsave_12/RestoreV2:25*
validate_shape(*
_output_shapes

:*
T0*
use_locking(*$
_class
loc:@pi/dense_3/kernel
�
save_12/Assign_26Assignpi/dense_3/kernel/Adamsave_12/RestoreV2:26*
validate_shape(*
T0*
_output_shapes

:*
use_locking(*$
_class
loc:@pi/dense_3/kernel
�
save_12/Assign_27Assignpi/dense_3/kernel/Adam_1save_12/RestoreV2:27*
_output_shapes

:*
T0*
validate_shape(*$
_class
loc:@pi/dense_3/kernel*
use_locking(
�
save_12/Assign_28Assignv/dense/biassave_12/RestoreV2:28*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(
�
save_12/Assign_29Assignv/dense/bias/Adamsave_12/RestoreV2:29*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@v/dense/bias
�
save_12/Assign_30Assignv/dense/bias/Adam_1save_12/RestoreV2:30*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
T0
�
save_12/Assign_31Assignv/dense/kernelsave_12/RestoreV2:31*
_output_shapes

: *
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
�
save_12/Assign_32Assignv/dense/kernel/Adamsave_12/RestoreV2:32*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes

: 
�
save_12/Assign_33Assignv/dense/kernel/Adam_1save_12/RestoreV2:33*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes

: 
�
save_12/Assign_34Assignv/dense_1/biassave_12/RestoreV2:34*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
use_locking(
�
save_12/Assign_35Assignv/dense_1/bias/Adamsave_12/RestoreV2:35*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_12/Assign_36Assignv/dense_1/bias/Adam_1save_12/RestoreV2:36*
_output_shapes
:*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(
�
save_12/Assign_37Assignv/dense_1/kernelsave_12/RestoreV2:37*
use_locking(*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0
�
save_12/Assign_38Assignv/dense_1/kernel/Adamsave_12/RestoreV2:38*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

: 
�
save_12/Assign_39Assignv/dense_1/kernel/Adam_1save_12/RestoreV2:39*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
validate_shape(
�
save_12/Assign_40Assignv/dense_2/biassave_12/RestoreV2:40*
use_locking(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_12/Assign_41Assignv/dense_2/bias/Adamsave_12/RestoreV2:41*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias
�
save_12/Assign_42Assignv/dense_2/bias/Adam_1save_12/RestoreV2:42*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_12/Assign_43Assignv/dense_2/kernelsave_12/RestoreV2:43*
_output_shapes

:*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel
�
save_12/Assign_44Assignv/dense_2/kernel/Adamsave_12/RestoreV2:44*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:
�
save_12/Assign_45Assignv/dense_2/kernel/Adam_1save_12/RestoreV2:45*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:
�
save_12/Assign_46Assignv/dense_3/biassave_12/RestoreV2:46*!
_class
loc:@v/dense_3/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_12/Assign_47Assignv/dense_3/bias/Adamsave_12/RestoreV2:47*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_3/bias*
T0*
use_locking(
�
save_12/Assign_48Assignv/dense_3/bias/Adam_1save_12/RestoreV2:48*!
_class
loc:@v/dense_3/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
�
save_12/Assign_49Assignv/dense_3/kernelsave_12/RestoreV2:49*
validate_shape(*
T0*#
_class
loc:@v/dense_3/kernel*
use_locking(*
_output_shapes

:
�
save_12/Assign_50Assignv/dense_3/kernel/Adamsave_12/RestoreV2:50*
use_locking(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
T0*
validate_shape(
�
save_12/Assign_51Assignv/dense_3/kernel/Adam_1save_12/RestoreV2:51*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@v/dense_3/kernel*
validate_shape(
�
save_12/Assign_52Assignv/dense_4/biassave_12/RestoreV2:52*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
use_locking(*
validate_shape(
�
save_12/Assign_53Assignv/dense_4/bias/Adamsave_12/RestoreV2:53*!
_class
loc:@v/dense_4/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
�
save_12/Assign_54Assignv/dense_4/bias/Adam_1save_12/RestoreV2:54*!
_class
loc:@v/dense_4/bias*
T0*
use_locking(*
_output_shapes
:@*
validate_shape(
�
save_12/Assign_55Assignv/dense_4/kernelsave_12/RestoreV2:55*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*
validate_shape(*
T0*
use_locking(
�
save_12/Assign_56Assignv/dense_4/kernel/Adamsave_12/RestoreV2:56*
T0*
use_locking(*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
validate_shape(
�
save_12/Assign_57Assignv/dense_4/kernel/Adam_1save_12/RestoreV2:57*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*
use_locking(*
T0
�
save_12/Assign_58Assignv/dense_5/biassave_12/RestoreV2:58*
validate_shape(*
use_locking(*
_output_shapes
: *!
_class
loc:@v/dense_5/bias*
T0
�
save_12/Assign_59Assignv/dense_5/bias/Adamsave_12/RestoreV2:59*
_output_shapes
: *
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_5/bias
�
save_12/Assign_60Assignv/dense_5/bias/Adam_1save_12/RestoreV2:60*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_5/bias*
_output_shapes
: 
�
save_12/Assign_61Assignv/dense_5/kernelsave_12/RestoreV2:61*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel*
T0*
use_locking(*
validate_shape(
�
save_12/Assign_62Assignv/dense_5/kernel/Adamsave_12/RestoreV2:62*
use_locking(*
validate_shape(*
T0*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel
�
save_12/Assign_63Assignv/dense_5/kernel/Adam_1save_12/RestoreV2:63*
use_locking(*
T0*#
_class
loc:@v/dense_5/kernel*
validate_shape(*
_output_shapes

:@ 
�
save_12/Assign_64Assignv/dense_6/biassave_12/RestoreV2:64*
use_locking(*!
_class
loc:@v/dense_6/bias*
validate_shape(*
_output_shapes
:*
T0
�
save_12/Assign_65Assignv/dense_6/bias/Adamsave_12/RestoreV2:65*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
validate_shape(
�
save_12/Assign_66Assignv/dense_6/bias/Adam_1save_12/RestoreV2:66*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_6/bias
�
save_12/Assign_67Assignv/dense_6/kernelsave_12/RestoreV2:67*
T0*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_6/kernel*
use_locking(
�
save_12/Assign_68Assignv/dense_6/kernel/Adamsave_12/RestoreV2:68*
validate_shape(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_6/kernel*
use_locking(
�
save_12/Assign_69Assignv/dense_6/kernel/Adam_1save_12/RestoreV2:69*
use_locking(*
_output_shapes

: *
T0*
validate_shape(*#
_class
loc:@v/dense_6/kernel
�
save_12/Assign_70Assignv/dense_7/biassave_12/RestoreV2:70*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_7/bias*
T0*
_output_shapes
:
�
save_12/Assign_71Assignv/dense_7/bias/Adamsave_12/RestoreV2:71*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_7/bias*
_output_shapes
:*
T0
�
save_12/Assign_72Assignv/dense_7/bias/Adam_1save_12/RestoreV2:72*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_7/bias
�
save_12/Assign_73Assignv/dense_7/kernelsave_12/RestoreV2:73*#
_class
loc:@v/dense_7/kernel*
_output_shapes

:*
validate_shape(*
use_locking(*
T0
�
save_12/Assign_74Assignv/dense_7/kernel/Adamsave_12/RestoreV2:74*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
use_locking(*
T0*
validate_shape(
�
save_12/Assign_75Assignv/dense_7/kernel/Adam_1save_12/RestoreV2:75*
T0*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_38^save_12/Assign_39^save_12/Assign_4^save_12/Assign_40^save_12/Assign_41^save_12/Assign_42^save_12/Assign_43^save_12/Assign_44^save_12/Assign_45^save_12/Assign_46^save_12/Assign_47^save_12/Assign_48^save_12/Assign_49^save_12/Assign_5^save_12/Assign_50^save_12/Assign_51^save_12/Assign_52^save_12/Assign_53^save_12/Assign_54^save_12/Assign_55^save_12/Assign_56^save_12/Assign_57^save_12/Assign_58^save_12/Assign_59^save_12/Assign_6^save_12/Assign_60^save_12/Assign_61^save_12/Assign_62^save_12/Assign_63^save_12/Assign_64^save_12/Assign_65^save_12/Assign_66^save_12/Assign_67^save_12/Assign_68^save_12/Assign_69^save_12/Assign_7^save_12/Assign_70^save_12/Assign_71^save_12/Assign_72^save_12/Assign_73^save_12/Assign_74^save_12/Assign_75^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_13/StringJoin/inputs_1Const*<
value3B1 B+_temp_3ca5cc9718724e3fbe1749d9f89faf51/part*
_output_shapes
: *
dtype0
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_13/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
_
save_13/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
�
save_13/SaveV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
save_13/SaveV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:L
�
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2**
_class 
loc:@save_13/ShardedFilename*
_output_shapes
: *
T0
�
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
N*
_output_shapes
:*
T0*

axis 
�
save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(
�
save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
T0*
_output_shapes
: 
�
save_13/RestoreV2/tensor_namesConst*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
_output_shapes
:L
�
"save_13/RestoreV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:L
�
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
validate_shape(*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
T0*
use_locking(
�
save_13/Assign_1Assignbeta1_power_1save_13/RestoreV2:1*
_class
loc:@v/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
�
save_13/Assign_2Assignbeta2_powersave_13/RestoreV2:2*
_output_shapes
: * 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(*
T0
�
save_13/Assign_3Assignbeta2_power_1save_13/RestoreV2:3*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(
�
save_13/Assign_4Assignpi/dense/biassave_13/RestoreV2:4*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
�
save_13/Assign_5Assignpi/dense/bias/Adamsave_13/RestoreV2:5*
_output_shapes
: *
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_13/Assign_6Assignpi/dense/bias/Adam_1save_13/RestoreV2:6*
validate_shape(*
T0*
use_locking(*
_output_shapes
: * 
_class
loc:@pi/dense/bias
�
save_13/Assign_7Assignpi/dense/kernelsave_13/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(
�
save_13/Assign_8Assignpi/dense/kernel/Adamsave_13/RestoreV2:8*
use_locking(*
validate_shape(*
_output_shapes

: *
T0*"
_class
loc:@pi/dense/kernel
�
save_13/Assign_9Assignpi/dense/kernel/Adam_1save_13/RestoreV2:9*
_output_shapes

: *
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
validate_shape(
�
save_13/Assign_10Assignpi/dense_1/biassave_13/RestoreV2:10*
T0*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(
�
save_13/Assign_11Assignpi/dense_1/bias/Adamsave_13/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_13/Assign_12Assignpi/dense_1/bias/Adam_1save_13/RestoreV2:12*
_output_shapes
:*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
T0*
use_locking(
�
save_13/Assign_13Assignpi/dense_1/kernelsave_13/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

: *
T0*
use_locking(
�
save_13/Assign_14Assignpi/dense_1/kernel/Adamsave_13/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
use_locking(*
_output_shapes

: *
validate_shape(*
T0
�
save_13/Assign_15Assignpi/dense_1/kernel/Adam_1save_13/RestoreV2:15*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes

: 
�
save_13/Assign_16Assignpi/dense_2/biassave_13/RestoreV2:16*
T0*
use_locking(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_13/Assign_17Assignpi/dense_2/bias/Adamsave_13/RestoreV2:17*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0
�
save_13/Assign_18Assignpi/dense_2/bias/Adam_1save_13/RestoreV2:18*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
T0*
_output_shapes
:
�
save_13/Assign_19Assignpi/dense_2/kernelsave_13/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:*
T0*
validate_shape(
�
save_13/Assign_20Assignpi/dense_2/kernel/Adamsave_13/RestoreV2:20*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:
�
save_13/Assign_21Assignpi/dense_2/kernel/Adam_1save_13/RestoreV2:21*
_output_shapes

:*
T0*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
�
save_13/Assign_22Assignpi/dense_3/biassave_13/RestoreV2:22*"
_class
loc:@pi/dense_3/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
�
save_13/Assign_23Assignpi/dense_3/bias/Adamsave_13/RestoreV2:23*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
T0*
_output_shapes
:*
use_locking(
�
save_13/Assign_24Assignpi/dense_3/bias/Adam_1save_13/RestoreV2:24*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
T0
�
save_13/Assign_25Assignpi/dense_3/kernelsave_13/RestoreV2:25*
T0*
_output_shapes

:*$
_class
loc:@pi/dense_3/kernel*
use_locking(*
validate_shape(
�
save_13/Assign_26Assignpi/dense_3/kernel/Adamsave_13/RestoreV2:26*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0*
use_locking(
�
save_13/Assign_27Assignpi/dense_3/kernel/Adam_1save_13/RestoreV2:27*
T0*
_output_shapes

:*
validate_shape(*$
_class
loc:@pi/dense_3/kernel*
use_locking(
�
save_13/Assign_28Assignv/dense/biassave_13/RestoreV2:28*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0*
validate_shape(*
use_locking(
�
save_13/Assign_29Assignv/dense/bias/Adamsave_13/RestoreV2:29*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
T0
�
save_13/Assign_30Assignv/dense/bias/Adam_1save_13/RestoreV2:30*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
�
save_13/Assign_31Assignv/dense/kernelsave_13/RestoreV2:31*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes

: 
�
save_13/Assign_32Assignv/dense/kernel/Adamsave_13/RestoreV2:32*
_output_shapes

: *
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_13/Assign_33Assignv/dense/kernel/Adam_1save_13/RestoreV2:33*
_output_shapes

: *!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
validate_shape(
�
save_13/Assign_34Assignv/dense_1/biassave_13/RestoreV2:34*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_13/Assign_35Assignv/dense_1/bias/Adamsave_13/RestoreV2:35*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
save_13/Assign_36Assignv/dense_1/bias/Adam_1save_13/RestoreV2:36*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_13/Assign_37Assignv/dense_1/kernelsave_13/RestoreV2:37*
_output_shapes

: *
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_13/Assign_38Assignv/dense_1/kernel/Adamsave_13/RestoreV2:38*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_13/Assign_39Assignv/dense_1/kernel/Adam_1save_13/RestoreV2:39*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

: 
�
save_13/Assign_40Assignv/dense_2/biassave_13/RestoreV2:40*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(*
_output_shapes
:
�
save_13/Assign_41Assignv/dense_2/bias/Adamsave_13/RestoreV2:41*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
�
save_13/Assign_42Assignv/dense_2/bias/Adam_1save_13/RestoreV2:42*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias
�
save_13/Assign_43Assignv/dense_2/kernelsave_13/RestoreV2:43*
use_locking(*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
T0
�
save_13/Assign_44Assignv/dense_2/kernel/Adamsave_13/RestoreV2:44*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
T0
�
save_13/Assign_45Assignv/dense_2/kernel/Adam_1save_13/RestoreV2:45*
_output_shapes

:*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(
�
save_13/Assign_46Assignv/dense_3/biassave_13/RestoreV2:46*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
use_locking(*
T0
�
save_13/Assign_47Assignv/dense_3/bias/Adamsave_13/RestoreV2:47*
validate_shape(*!
_class
loc:@v/dense_3/bias*
T0*
_output_shapes
:*
use_locking(
�
save_13/Assign_48Assignv/dense_3/bias/Adam_1save_13/RestoreV2:48*
T0*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
validate_shape(*
use_locking(
�
save_13/Assign_49Assignv/dense_3/kernelsave_13/RestoreV2:49*
T0*
validate_shape(*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel
�
save_13/Assign_50Assignv/dense_3/kernel/Adamsave_13/RestoreV2:50*#
_class
loc:@v/dense_3/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:
�
save_13/Assign_51Assignv/dense_3/kernel/Adam_1save_13/RestoreV2:51*
T0*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
use_locking(
�
save_13/Assign_52Assignv/dense_4/biassave_13/RestoreV2:52*
use_locking(*!
_class
loc:@v/dense_4/bias*
T0*
_output_shapes
:@*
validate_shape(
�
save_13/Assign_53Assignv/dense_4/bias/Adamsave_13/RestoreV2:53*
use_locking(*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
T0*
validate_shape(
�
save_13/Assign_54Assignv/dense_4/bias/Adam_1save_13/RestoreV2:54*
T0*
validate_shape(*
_output_shapes
:@*
use_locking(*!
_class
loc:@v/dense_4/bias
�
save_13/Assign_55Assignv/dense_4/kernelsave_13/RestoreV2:55*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@
�
save_13/Assign_56Assignv/dense_4/kernel/Adamsave_13/RestoreV2:56*
_output_shapes
:	�@*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
T0*
use_locking(
�
save_13/Assign_57Assignv/dense_4/kernel/Adam_1save_13/RestoreV2:57*
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@
�
save_13/Assign_58Assignv/dense_5/biassave_13/RestoreV2:58*
use_locking(*
_output_shapes
: *
T0*!
_class
loc:@v/dense_5/bias*
validate_shape(
�
save_13/Assign_59Assignv/dense_5/bias/Adamsave_13/RestoreV2:59*!
_class
loc:@v/dense_5/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_13/Assign_60Assignv/dense_5/bias/Adam_1save_13/RestoreV2:60*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_5/bias
�
save_13/Assign_61Assignv/dense_5/kernelsave_13/RestoreV2:61*
_output_shapes

:@ *
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_5/kernel
�
save_13/Assign_62Assignv/dense_5/kernel/Adamsave_13/RestoreV2:62*
use_locking(*
T0*#
_class
loc:@v/dense_5/kernel*
_output_shapes

:@ *
validate_shape(
�
save_13/Assign_63Assignv/dense_5/kernel/Adam_1save_13/RestoreV2:63*
validate_shape(*
_output_shapes

:@ *#
_class
loc:@v/dense_5/kernel*
use_locking(*
T0
�
save_13/Assign_64Assignv/dense_6/biassave_13/RestoreV2:64*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_6/bias*
use_locking(
�
save_13/Assign_65Assignv/dense_6/bias/Adamsave_13/RestoreV2:65*!
_class
loc:@v/dense_6/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_13/Assign_66Assignv/dense_6/bias/Adam_1save_13/RestoreV2:66*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_6/bias*
T0*
_output_shapes
:
�
save_13/Assign_67Assignv/dense_6/kernelsave_13/RestoreV2:67*
validate_shape(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_6/kernel*
use_locking(
�
save_13/Assign_68Assignv/dense_6/kernel/Adamsave_13/RestoreV2:68*
_output_shapes

: *
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_6/kernel
�
save_13/Assign_69Assignv/dense_6/kernel/Adam_1save_13/RestoreV2:69*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*#
_class
loc:@v/dense_6/kernel
�
save_13/Assign_70Assignv/dense_7/biassave_13/RestoreV2:70*!
_class
loc:@v/dense_7/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_13/Assign_71Assignv/dense_7/bias/Adamsave_13/RestoreV2:71*
use_locking(*!
_class
loc:@v/dense_7/bias*
T0*
_output_shapes
:*
validate_shape(
�
save_13/Assign_72Assignv/dense_7/bias/Adam_1save_13/RestoreV2:72*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_7/bias*
T0*
validate_shape(
�
save_13/Assign_73Assignv/dense_7/kernelsave_13/RestoreV2:73*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_7/kernel*
T0*
_output_shapes

:
�
save_13/Assign_74Assignv/dense_7/kernel/Adamsave_13/RestoreV2:74*
use_locking(*
T0*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
_output_shapes

:
�
save_13/Assign_75Assignv/dense_7/kernel/Adam_1save_13/RestoreV2:75*
T0*
_output_shapes

:*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_7/kernel
�
save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_38^save_13/Assign_39^save_13/Assign_4^save_13/Assign_40^save_13/Assign_41^save_13/Assign_42^save_13/Assign_43^save_13/Assign_44^save_13/Assign_45^save_13/Assign_46^save_13/Assign_47^save_13/Assign_48^save_13/Assign_49^save_13/Assign_5^save_13/Assign_50^save_13/Assign_51^save_13/Assign_52^save_13/Assign_53^save_13/Assign_54^save_13/Assign_55^save_13/Assign_56^save_13/Assign_57^save_13/Assign_58^save_13/Assign_59^save_13/Assign_6^save_13/Assign_60^save_13/Assign_61^save_13/Assign_62^save_13/Assign_63^save_13/Assign_64^save_13/Assign_65^save_13/Assign_66^save_13/Assign_67^save_13/Assign_68^save_13/Assign_69^save_13/Assign_7^save_13/Assign_70^save_13/Assign_71^save_13/Assign_72^save_13/Assign_73^save_13/Assign_74^save_13/Assign_75^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
_output_shapes
: *
dtype0*
shape: 
�
save_14/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_0d028a70038745a98f9a36abc0c7d1a1/part
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_14/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_14/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
�
save_14/SaveV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
save_14/SaveV2/shape_and_slicesConst*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L
�
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
T0**
_class 
loc:@save_14/ShardedFilename*
_output_shapes
: 
�
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*
T0*

axis *
_output_shapes
:*
N
�
save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(
�
save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
T0*
_output_shapes
: 
�
save_14/RestoreV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
"save_14/RestoreV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L
�
save_14/AssignAssignbeta1_powersave_14/RestoreV2*
_output_shapes
: *
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_14/Assign_1Assignbeta1_power_1save_14/RestoreV2:1*
validate_shape(*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias
�
save_14/Assign_2Assignbeta2_powersave_14/RestoreV2:2*
use_locking(*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_14/Assign_3Assignbeta2_power_1save_14/RestoreV2:3*
_output_shapes
: *
T0*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_14/Assign_4Assignpi/dense/biassave_14/RestoreV2:4*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias*
T0*
_output_shapes
: 
�
save_14/Assign_5Assignpi/dense/bias/Adamsave_14/RestoreV2:5*
use_locking(*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
T0
�
save_14/Assign_6Assignpi/dense/bias/Adam_1save_14/RestoreV2:6*
T0*
_output_shapes
: *
validate_shape(* 
_class
loc:@pi/dense/bias*
use_locking(
�
save_14/Assign_7Assignpi/dense/kernelsave_14/RestoreV2:7*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel*
_output_shapes

: 
�
save_14/Assign_8Assignpi/dense/kernel/Adamsave_14/RestoreV2:8*
_output_shapes

: *
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi/dense/kernel
�
save_14/Assign_9Assignpi/dense/kernel/Adam_1save_14/RestoreV2:9*
validate_shape(*
use_locking(*
T0*
_output_shapes

: *"
_class
loc:@pi/dense/kernel
�
save_14/Assign_10Assignpi/dense_1/biassave_14/RestoreV2:10*
T0*
use_locking(*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
validate_shape(
�
save_14/Assign_11Assignpi/dense_1/bias/Adamsave_14/RestoreV2:11*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_1/bias
�
save_14/Assign_12Assignpi/dense_1/bias/Adam_1save_14/RestoreV2:12*
_output_shapes
:*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
use_locking(
�
save_14/Assign_13Assignpi/dense_1/kernelsave_14/RestoreV2:13*
_output_shapes

: *$
_class
loc:@pi/dense_1/kernel*
use_locking(*
validate_shape(*
T0
�
save_14/Assign_14Assignpi/dense_1/kernel/Adamsave_14/RestoreV2:14*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: 
�
save_14/Assign_15Assignpi/dense_1/kernel/Adam_1save_14/RestoreV2:15*
_output_shapes

: *
use_locking(*$
_class
loc:@pi/dense_1/kernel*
T0*
validate_shape(
�
save_14/Assign_16Assignpi/dense_2/biassave_14/RestoreV2:16*
use_locking(*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
�
save_14/Assign_17Assignpi/dense_2/bias/Adamsave_14/RestoreV2:17*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
�
save_14/Assign_18Assignpi/dense_2/bias/Adam_1save_14/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
�
save_14/Assign_19Assignpi/dense_2/kernelsave_14/RestoreV2:19*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:
�
save_14/Assign_20Assignpi/dense_2/kernel/Adamsave_14/RestoreV2:20*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
_output_shapes

:*
T0
�
save_14/Assign_21Assignpi/dense_2/kernel/Adam_1save_14/RestoreV2:21*
validate_shape(*
_output_shapes

:*
T0*$
_class
loc:@pi/dense_2/kernel*
use_locking(
�
save_14/Assign_22Assignpi/dense_3/biassave_14/RestoreV2:22*
T0*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
_output_shapes
:*
use_locking(
�
save_14/Assign_23Assignpi/dense_3/bias/Adamsave_14/RestoreV2:23*
T0*"
_class
loc:@pi/dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_14/Assign_24Assignpi/dense_3/bias/Adam_1save_14/RestoreV2:24*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
validate_shape(*
use_locking(*
T0
�
save_14/Assign_25Assignpi/dense_3/kernelsave_14/RestoreV2:25*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:
�
save_14/Assign_26Assignpi/dense_3/kernel/Adamsave_14/RestoreV2:26*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:
�
save_14/Assign_27Assignpi/dense_3/kernel/Adam_1save_14/RestoreV2:27*
T0*$
_class
loc:@pi/dense_3/kernel*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_14/Assign_28Assignv/dense/biassave_14/RestoreV2:28*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0*
use_locking(*
validate_shape(
�
save_14/Assign_29Assignv/dense/bias/Adamsave_14/RestoreV2:29*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
T0
�
save_14/Assign_30Assignv/dense/bias/Adam_1save_14/RestoreV2:30*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_14/Assign_31Assignv/dense/kernelsave_14/RestoreV2:31*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes

: *
validate_shape(
�
save_14/Assign_32Assignv/dense/kernel/Adamsave_14/RestoreV2:32*
_output_shapes

: *
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_14/Assign_33Assignv/dense/kernel/Adam_1save_14/RestoreV2:33*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes

: *
use_locking(
�
save_14/Assign_34Assignv/dense_1/biassave_14/RestoreV2:34*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_1/bias
�
save_14/Assign_35Assignv/dense_1/bias/Adamsave_14/RestoreV2:35*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(
�
save_14/Assign_36Assignv/dense_1/bias/Adam_1save_14/RestoreV2:36*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:*
T0
�
save_14/Assign_37Assignv/dense_1/kernelsave_14/RestoreV2:37*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

: 
�
save_14/Assign_38Assignv/dense_1/kernel/Adamsave_14/RestoreV2:38*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
T0*
use_locking(
�
save_14/Assign_39Assignv/dense_1/kernel/Adam_1save_14/RestoreV2:39*
validate_shape(*
_output_shapes

: *
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0
�
save_14/Assign_40Assignv/dense_2/biassave_14/RestoreV2:40*
T0*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:
�
save_14/Assign_41Assignv/dense_2/bias/Adamsave_14/RestoreV2:41*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0
�
save_14/Assign_42Assignv/dense_2/bias/Adam_1save_14/RestoreV2:42*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_14/Assign_43Assignv/dense_2/kernelsave_14/RestoreV2:43*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_14/Assign_44Assignv/dense_2/kernel/Adamsave_14/RestoreV2:44*#
_class
loc:@v/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_14/Assign_45Assignv/dense_2/kernel/Adam_1save_14/RestoreV2:45*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_14/Assign_46Assignv/dense_3/biassave_14/RestoreV2:46*
use_locking(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_3/bias*
validate_shape(
�
save_14/Assign_47Assignv/dense_3/bias/Adamsave_14/RestoreV2:47*!
_class
loc:@v/dense_3/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_14/Assign_48Assignv/dense_3/bias/Adam_1save_14/RestoreV2:48*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
T0
�
save_14/Assign_49Assignv/dense_3/kernelsave_14/RestoreV2:49*
_output_shapes

:*
use_locking(*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
T0
�
save_14/Assign_50Assignv/dense_3/kernel/Adamsave_14/RestoreV2:50*
T0*
validate_shape(*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel
�
save_14/Assign_51Assignv/dense_3/kernel/Adam_1save_14/RestoreV2:51*
T0*
_output_shapes

:*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_3/kernel
�
save_14/Assign_52Assignv/dense_4/biassave_14/RestoreV2:52*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
validate_shape(*
use_locking(
�
save_14/Assign_53Assignv/dense_4/bias/Adamsave_14/RestoreV2:53*
_output_shapes
:@*
T0*
validate_shape(*!
_class
loc:@v/dense_4/bias*
use_locking(
�
save_14/Assign_54Assignv/dense_4/bias/Adam_1save_14/RestoreV2:54*!
_class
loc:@v/dense_4/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:@
�
save_14/Assign_55Assignv/dense_4/kernelsave_14/RestoreV2:55*#
_class
loc:@v/dense_4/kernel*
validate_shape(*
T0*
_output_shapes
:	�@*
use_locking(
�
save_14/Assign_56Assignv/dense_4/kernel/Adamsave_14/RestoreV2:56*
validate_shape(*
_output_shapes
:	�@*
use_locking(*
T0*#
_class
loc:@v/dense_4/kernel
�
save_14/Assign_57Assignv/dense_4/kernel/Adam_1save_14/RestoreV2:57*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@*
T0
�
save_14/Assign_58Assignv/dense_5/biassave_14/RestoreV2:58*
T0*
use_locking(*!
_class
loc:@v/dense_5/bias*
_output_shapes
: *
validate_shape(
�
save_14/Assign_59Assignv/dense_5/bias/Adamsave_14/RestoreV2:59*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_5/bias*
_output_shapes
: 
�
save_14/Assign_60Assignv/dense_5/bias/Adam_1save_14/RestoreV2:60*
_output_shapes
: *
use_locking(*
validate_shape(*!
_class
loc:@v/dense_5/bias*
T0
�
save_14/Assign_61Assignv/dense_5/kernelsave_14/RestoreV2:61*
_output_shapes

:@ *
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_5/kernel
�
save_14/Assign_62Assignv/dense_5/kernel/Adamsave_14/RestoreV2:62*
T0*
_output_shapes

:@ *
use_locking(*
validate_shape(*#
_class
loc:@v/dense_5/kernel
�
save_14/Assign_63Assignv/dense_5/kernel/Adam_1save_14/RestoreV2:63*
_output_shapes

:@ *
use_locking(*#
_class
loc:@v/dense_5/kernel*
validate_shape(*
T0
�
save_14/Assign_64Assignv/dense_6/biassave_14/RestoreV2:64*!
_class
loc:@v/dense_6/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_14/Assign_65Assignv/dense_6/bias/Adamsave_14/RestoreV2:65*
T0*!
_class
loc:@v/dense_6/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_14/Assign_66Assignv/dense_6/bias/Adam_1save_14/RestoreV2:66*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_6/bias*
_output_shapes
:
�
save_14/Assign_67Assignv/dense_6/kernelsave_14/RestoreV2:67*
use_locking(*
_output_shapes

: *
T0*#
_class
loc:@v/dense_6/kernel*
validate_shape(
�
save_14/Assign_68Assignv/dense_6/kernel/Adamsave_14/RestoreV2:68*
T0*
use_locking(*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: *
validate_shape(
�
save_14/Assign_69Assignv/dense_6/kernel/Adam_1save_14/RestoreV2:69*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_6/kernel*
_output_shapes

: 
�
save_14/Assign_70Assignv/dense_7/biassave_14/RestoreV2:70*!
_class
loc:@v/dense_7/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_14/Assign_71Assignv/dense_7/bias/Adamsave_14/RestoreV2:71*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_7/bias*
T0
�
save_14/Assign_72Assignv/dense_7/bias/Adam_1save_14/RestoreV2:72*!
_class
loc:@v/dense_7/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
�
save_14/Assign_73Assignv/dense_7/kernelsave_14/RestoreV2:73*
validate_shape(*#
_class
loc:@v/dense_7/kernel*
use_locking(*
_output_shapes

:*
T0
�
save_14/Assign_74Assignv/dense_7/kernel/Adamsave_14/RestoreV2:74*
validate_shape(*#
_class
loc:@v/dense_7/kernel*
use_locking(*
_output_shapes

:*
T0
�
save_14/Assign_75Assignv/dense_7/kernel/Adam_1save_14/RestoreV2:75*
_output_shapes

:*
validate_shape(*
T0*#
_class
loc:@v/dense_7/kernel*
use_locking(
�
save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_38^save_14/Assign_39^save_14/Assign_4^save_14/Assign_40^save_14/Assign_41^save_14/Assign_42^save_14/Assign_43^save_14/Assign_44^save_14/Assign_45^save_14/Assign_46^save_14/Assign_47^save_14/Assign_48^save_14/Assign_49^save_14/Assign_5^save_14/Assign_50^save_14/Assign_51^save_14/Assign_52^save_14/Assign_53^save_14/Assign_54^save_14/Assign_55^save_14/Assign_56^save_14/Assign_57^save_14/Assign_58^save_14/Assign_59^save_14/Assign_6^save_14/Assign_60^save_14/Assign_61^save_14/Assign_62^save_14/Assign_63^save_14/Assign_64^save_14/Assign_65^save_14/Assign_66^save_14/Assign_67^save_14/Assign_68^save_14/Assign_69^save_14/Assign_7^save_14/Assign_70^save_14/Assign_71^save_14/Assign_72^save_14/Assign_73^save_14/Assign_74^save_14/Assign_75^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
dtype0*
shape: *
_output_shapes
: 
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
_output_shapes
: *
shape: *
dtype0
�
save_15/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_9d3650d029654d4f9c4b7b9aef93a5b2/part*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
T
save_15/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
_
save_15/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
�
save_15/SaveV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1
�
save_15/SaveV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1pi/dense_3/biaspi/dense_3/bias/Adampi/dense_3/bias/Adam_1pi/dense_3/kernelpi/dense_3/kernel/Adampi/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1v/dense_4/biasv/dense_4/bias/Adamv/dense_4/bias/Adam_1v/dense_4/kernelv/dense_4/kernel/Adamv/dense_4/kernel/Adam_1v/dense_5/biasv/dense_5/bias/Adamv/dense_5/bias/Adam_1v/dense_5/kernelv/dense_5/kernel/Adamv/dense_5/kernel/Adam_1v/dense_6/biasv/dense_6/bias/Adamv/dense_6/bias/Adam_1v/dense_6/kernelv/dense_6/kernel/Adamv/dense_6/kernel/Adam_1v/dense_7/biasv/dense_7/bias/Adamv/dense_7/bias/Adam_1v/dense_7/kernelv/dense_7/kernel/Adamv/dense_7/kernel/Adam_1*Z
dtypesP
N2L
�
save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2*
_output_shapes
: **
_class 
loc:@save_15/ShardedFilename*
T0
�
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*
_output_shapes
:*

axis *
N*
T0
�
save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(
�
save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
T0*
_output_shapes
: 
�
save_15/RestoreV2/tensor_namesConst*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bpi/dense_3/biasBpi/dense_3/bias/AdamBpi/dense_3/bias/Adam_1Bpi/dense_3/kernelBpi/dense_3/kernel/AdamBpi/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1Bv/dense_4/biasBv/dense_4/bias/AdamBv/dense_4/bias/Adam_1Bv/dense_4/kernelBv/dense_4/kernel/AdamBv/dense_4/kernel/Adam_1Bv/dense_5/biasBv/dense_5/bias/AdamBv/dense_5/bias/Adam_1Bv/dense_5/kernelBv/dense_5/kernel/AdamBv/dense_5/kernel/Adam_1Bv/dense_6/biasBv/dense_6/bias/AdamBv/dense_6/bias/Adam_1Bv/dense_6/kernelBv/dense_6/kernel/AdamBv/dense_6/kernel/Adam_1Bv/dense_7/biasBv/dense_7/bias/AdamBv/dense_7/bias/Adam_1Bv/dense_7/kernelBv/dense_7/kernel/AdamBv/dense_7/kernel/Adam_1*
_output_shapes
:L
�
"save_15/RestoreV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L
�
save_15/AssignAssignbeta1_powersave_15/RestoreV2*
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
�
save_15/Assign_1Assignbeta1_power_1save_15/RestoreV2:1*
_class
loc:@v/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
�
save_15/Assign_2Assignbeta2_powersave_15/RestoreV2:2* 
_class
loc:@pi/dense/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_15/Assign_3Assignbeta2_power_1save_15/RestoreV2:3*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@v/dense/bias
�
save_15/Assign_4Assignpi/dense/biassave_15/RestoreV2:4*
_output_shapes
: *
T0*
use_locking(*
validate_shape(* 
_class
loc:@pi/dense/bias
�
save_15/Assign_5Assignpi/dense/bias/Adamsave_15/RestoreV2:5* 
_class
loc:@pi/dense/bias*
_output_shapes
: *
use_locking(*
T0*
validate_shape(
�
save_15/Assign_6Assignpi/dense/bias/Adam_1save_15/RestoreV2:6* 
_class
loc:@pi/dense/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
�
save_15/Assign_7Assignpi/dense/kernelsave_15/RestoreV2:7*
_output_shapes

: *
use_locking(*
T0*
validate_shape(*"
_class
loc:@pi/dense/kernel
�
save_15/Assign_8Assignpi/dense/kernel/Adamsave_15/RestoreV2:8*
validate_shape(*
use_locking(*
_output_shapes

: *"
_class
loc:@pi/dense/kernel*
T0
�
save_15/Assign_9Assignpi/dense/kernel/Adam_1save_15/RestoreV2:9*
T0*
_output_shapes

: *
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel
�
save_15/Assign_10Assignpi/dense_1/biassave_15/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:*
validate_shape(
�
save_15/Assign_11Assignpi/dense_1/bias/Adamsave_15/RestoreV2:11*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_1/bias*
use_locking(*
T0
�
save_15/Assign_12Assignpi/dense_1/bias/Adam_1save_15/RestoreV2:12*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_1/bias
�
save_15/Assign_13Assignpi/dense_1/kernelsave_15/RestoreV2:13*
use_locking(*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

: 
�
save_15/Assign_14Assignpi/dense_1/kernel/Adamsave_15/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: *
validate_shape(
�
save_15/Assign_15Assignpi/dense_1/kernel/Adam_1save_15/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(
�
save_15/Assign_16Assignpi/dense_2/biassave_15/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_15/Assign_17Assignpi/dense_2/bias/Adamsave_15/RestoreV2:17*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
�
save_15/Assign_18Assignpi/dense_2/bias/Adam_1save_15/RestoreV2:18*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*"
_class
loc:@pi/dense_2/bias
�
save_15/Assign_19Assignpi/dense_2/kernelsave_15/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
�
save_15/Assign_20Assignpi/dense_2/kernel/Adamsave_15/RestoreV2:20*
validate_shape(*
T0*
use_locking(*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel
�
save_15/Assign_21Assignpi/dense_2/kernel/Adam_1save_15/RestoreV2:21*
use_locking(*
T0*
_output_shapes

:*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
�
save_15/Assign_22Assignpi/dense_3/biassave_15/RestoreV2:22*
validate_shape(*"
_class
loc:@pi/dense_3/bias*
use_locking(*
_output_shapes
:*
T0
�
save_15/Assign_23Assignpi/dense_3/bias/Adamsave_15/RestoreV2:23*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
T0*
validate_shape(
�
save_15/Assign_24Assignpi/dense_3/bias/Adam_1save_15/RestoreV2:24*
_output_shapes
:*"
_class
loc:@pi/dense_3/bias*
use_locking(*
T0*
validate_shape(
�
save_15/Assign_25Assignpi/dense_3/kernelsave_15/RestoreV2:25*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0*
use_locking(
�
save_15/Assign_26Assignpi/dense_3/kernel/Adamsave_15/RestoreV2:26*
_output_shapes

:*
validate_shape(*
T0*$
_class
loc:@pi/dense_3/kernel*
use_locking(
�
save_15/Assign_27Assignpi/dense_3/kernel/Adam_1save_15/RestoreV2:27*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_3/kernel*
_output_shapes

:*
T0
�
save_15/Assign_28Assignv/dense/biassave_15/RestoreV2:28*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
�
save_15/Assign_29Assignv/dense/bias/Adamsave_15/RestoreV2:29*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
�
save_15/Assign_30Assignv/dense/bias/Adam_1save_15/RestoreV2:30*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_15/Assign_31Assignv/dense/kernelsave_15/RestoreV2:31*
validate_shape(*
_output_shapes

: *!
_class
loc:@v/dense/kernel*
use_locking(*
T0
�
save_15/Assign_32Assignv/dense/kernel/Adamsave_15/RestoreV2:32*
T0*
use_locking(*
_output_shapes

: *
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_15/Assign_33Assignv/dense/kernel/Adam_1save_15/RestoreV2:33*
_output_shapes

: *
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(*
T0
�
save_15/Assign_34Assignv/dense_1/biassave_15/RestoreV2:34*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_15/Assign_35Assignv/dense_1/bias/Adamsave_15/RestoreV2:35*
T0*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(
�
save_15/Assign_36Assignv/dense_1/bias/Adam_1save_15/RestoreV2:36*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
�
save_15/Assign_37Assignv/dense_1/kernelsave_15/RestoreV2:37*
_output_shapes

: *
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel
�
save_15/Assign_38Assignv/dense_1/kernel/Adamsave_15/RestoreV2:38*
_output_shapes

: *
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_1/kernel
�
save_15/Assign_39Assignv/dense_1/kernel/Adam_1save_15/RestoreV2:39*
_output_shapes

: *
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
�
save_15/Assign_40Assignv/dense_2/biassave_15/RestoreV2:40*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:
�
save_15/Assign_41Assignv/dense_2/bias/Adamsave_15/RestoreV2:41*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_15/Assign_42Assignv/dense_2/bias/Adam_1save_15/RestoreV2:42*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_15/Assign_43Assignv/dense_2/kernelsave_15/RestoreV2:43*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
T0*
use_locking(*
validate_shape(
�
save_15/Assign_44Assignv/dense_2/kernel/Adamsave_15/RestoreV2:44*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_15/Assign_45Assignv/dense_2/kernel/Adam_1save_15/RestoreV2:45*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes

:*
T0
�
save_15/Assign_46Assignv/dense_3/biassave_15/RestoreV2:46*
T0*
validate_shape(*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
use_locking(
�
save_15/Assign_47Assignv/dense_3/bias/Adamsave_15/RestoreV2:47*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias
�
save_15/Assign_48Assignv/dense_3/bias/Adam_1save_15/RestoreV2:48*
use_locking(*
T0*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
validate_shape(
�
save_15/Assign_49Assignv/dense_3/kernelsave_15/RestoreV2:49*
validate_shape(*
T0*
_output_shapes

:*
use_locking(*#
_class
loc:@v/dense_3/kernel
�
save_15/Assign_50Assignv/dense_3/kernel/Adamsave_15/RestoreV2:50*#
_class
loc:@v/dense_3/kernel*
use_locking(*
_output_shapes

:*
T0*
validate_shape(
�
save_15/Assign_51Assignv/dense_3/kernel/Adam_1save_15/RestoreV2:51*
validate_shape(*
_output_shapes

:*
use_locking(*#
_class
loc:@v/dense_3/kernel*
T0
�
save_15/Assign_52Assignv/dense_4/biassave_15/RestoreV2:52*!
_class
loc:@v/dense_4/bias*
_output_shapes
:@*
use_locking(*
T0*
validate_shape(
�
save_15/Assign_53Assignv/dense_4/bias/Adamsave_15/RestoreV2:53*
T0*
_output_shapes
:@*!
_class
loc:@v/dense_4/bias*
validate_shape(*
use_locking(
�
save_15/Assign_54Assignv/dense_4/bias/Adam_1save_15/RestoreV2:54*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_4/bias*
_output_shapes
:@
�
save_15/Assign_55Assignv/dense_4/kernelsave_15/RestoreV2:55*
T0*
use_locking(*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
validate_shape(
�
save_15/Assign_56Assignv/dense_4/kernel/Adamsave_15/RestoreV2:56*
_output_shapes
:	�@*#
_class
loc:@v/dense_4/kernel*
use_locking(*
T0*
validate_shape(
�
save_15/Assign_57Assignv/dense_4/kernel/Adam_1save_15/RestoreV2:57*
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_4/kernel*
_output_shapes
:	�@
�
save_15/Assign_58Assignv/dense_5/biassave_15/RestoreV2:58*!
_class
loc:@v/dense_5/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_15/Assign_59Assignv/dense_5/bias/Adamsave_15/RestoreV2:59*
_output_shapes
: *!
_class
loc:@v/dense_5/bias*
T0*
use_locking(*
validate_shape(
�
save_15/Assign_60Assignv/dense_5/bias/Adam_1save_15/RestoreV2:60*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*!
_class
loc:@v/dense_5/bias
�
save_15/Assign_61Assignv/dense_5/kernelsave_15/RestoreV2:61*
_output_shapes

:@ *
validate_shape(*
T0*#
_class
loc:@v/dense_5/kernel*
use_locking(
�
save_15/Assign_62Assignv/dense_5/kernel/Adamsave_15/RestoreV2:62*
_output_shapes

:@ *
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_5/kernel
�
save_15/Assign_63Assignv/dense_5/kernel/Adam_1save_15/RestoreV2:63*
_output_shapes

:@ *
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_5/kernel
�
save_15/Assign_64Assignv/dense_6/biassave_15/RestoreV2:64*
validate_shape(*!
_class
loc:@v/dense_6/bias*
use_locking(*
T0*
_output_shapes
:
�
save_15/Assign_65Assignv/dense_6/bias/Adamsave_15/RestoreV2:65*!
_class
loc:@v/dense_6/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_15/Assign_66Assignv/dense_6/bias/Adam_1save_15/RestoreV2:66*
_output_shapes
:*!
_class
loc:@v/dense_6/bias*
use_locking(*
T0*
validate_shape(
�
save_15/Assign_67Assignv/dense_6/kernelsave_15/RestoreV2:67*
_output_shapes

: *
validate_shape(*#
_class
loc:@v/dense_6/kernel*
T0*
use_locking(
�
save_15/Assign_68Assignv/dense_6/kernel/Adamsave_15/RestoreV2:68*
T0*#
_class
loc:@v/dense_6/kernel*
use_locking(*
validate_shape(*
_output_shapes

: 
�
save_15/Assign_69Assignv/dense_6/kernel/Adam_1save_15/RestoreV2:69*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*#
_class
loc:@v/dense_6/kernel
�
save_15/Assign_70Assignv/dense_7/biassave_15/RestoreV2:70*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense_7/bias
�
save_15/Assign_71Assignv/dense_7/bias/Adamsave_15/RestoreV2:71*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_7/bias
�
save_15/Assign_72Assignv/dense_7/bias/Adam_1save_15/RestoreV2:72*!
_class
loc:@v/dense_7/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
�
save_15/Assign_73Assignv/dense_7/kernelsave_15/RestoreV2:73*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
use_locking(*
validate_shape(*
T0
�
save_15/Assign_74Assignv/dense_7/kernel/Adamsave_15/RestoreV2:74*
_output_shapes

:*#
_class
loc:@v/dense_7/kernel*
validate_shape(*
T0*
use_locking(
�
save_15/Assign_75Assignv/dense_7/kernel/Adam_1save_15/RestoreV2:75*#
_class
loc:@v/dense_7/kernel*
_output_shapes

:*
use_locking(*
T0*
validate_shape(
�
save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_20^save_15/Assign_21^save_15/Assign_22^save_15/Assign_23^save_15/Assign_24^save_15/Assign_25^save_15/Assign_26^save_15/Assign_27^save_15/Assign_28^save_15/Assign_29^save_15/Assign_3^save_15/Assign_30^save_15/Assign_31^save_15/Assign_32^save_15/Assign_33^save_15/Assign_34^save_15/Assign_35^save_15/Assign_36^save_15/Assign_37^save_15/Assign_38^save_15/Assign_39^save_15/Assign_4^save_15/Assign_40^save_15/Assign_41^save_15/Assign_42^save_15/Assign_43^save_15/Assign_44^save_15/Assign_45^save_15/Assign_46^save_15/Assign_47^save_15/Assign_48^save_15/Assign_49^save_15/Assign_5^save_15/Assign_50^save_15/Assign_51^save_15/Assign_52^save_15/Assign_53^save_15/Assign_54^save_15/Assign_55^save_15/Assign_56^save_15/Assign_57^save_15/Assign_58^save_15/Assign_59^save_15/Assign_6^save_15/Assign_60^save_15/Assign_61^save_15/Assign_62^save_15/Assign_63^save_15/Assign_64^save_15/Assign_65^save_15/Assign_66^save_15/Assign_67^save_15/Assign_68^save_15/Assign_69^save_15/Assign_7^save_15/Assign_70^save_15/Assign_71^save_15/Assign_72^save_15/Assign_73^save_15/Assign_74^save_15/Assign_75^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard "&E
save_15/Const:0save_15/Identity:0save_15/restore_all (5 @F8"
train_v


Adam_1"�I
	variables�H�H
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
{
pi/dense_3/kernel:0pi/dense_3/kernel/Assignpi/dense_3/kernel/read:02.pi/dense_3/kernel/Initializer/random_uniform:08
j
pi/dense_3/bias:0pi/dense_3/bias/Assignpi/dense_3/bias/read:02#pi/dense_3/bias/Initializer/zeros:08
o
v/dense/kernel:0v/dense/kernel/Assignv/dense/kernel/read:02+v/dense/kernel/Initializer/random_uniform:08
^
v/dense/bias:0v/dense/bias/Assignv/dense/bias/read:02 v/dense/bias/Initializer/zeros:08
w
v/dense_1/kernel:0v/dense_1/kernel/Assignv/dense_1/kernel/read:02-v/dense_1/kernel/Initializer/random_uniform:08
f
v/dense_1/bias:0v/dense_1/bias/Assignv/dense_1/bias/read:02"v/dense_1/bias/Initializer/zeros:08
w
v/dense_2/kernel:0v/dense_2/kernel/Assignv/dense_2/kernel/read:02-v/dense_2/kernel/Initializer/random_uniform:08
f
v/dense_2/bias:0v/dense_2/bias/Assignv/dense_2/bias/read:02"v/dense_2/bias/Initializer/zeros:08
w
v/dense_3/kernel:0v/dense_3/kernel/Assignv/dense_3/kernel/read:02-v/dense_3/kernel/Initializer/random_uniform:08
f
v/dense_3/bias:0v/dense_3/bias/Assignv/dense_3/bias/read:02"v/dense_3/bias/Initializer/zeros:08
w
v/dense_4/kernel:0v/dense_4/kernel/Assignv/dense_4/kernel/read:02-v/dense_4/kernel/Initializer/random_uniform:08
f
v/dense_4/bias:0v/dense_4/bias/Assignv/dense_4/bias/read:02"v/dense_4/bias/Initializer/zeros:08
w
v/dense_5/kernel:0v/dense_5/kernel/Assignv/dense_5/kernel/read:02-v/dense_5/kernel/Initializer/random_uniform:08
f
v/dense_5/bias:0v/dense_5/bias/Assignv/dense_5/bias/read:02"v/dense_5/bias/Initializer/zeros:08
w
v/dense_6/kernel:0v/dense_6/kernel/Assignv/dense_6/kernel/read:02-v/dense_6/kernel/Initializer/random_uniform:08
f
v/dense_6/bias:0v/dense_6/bias/Assignv/dense_6/bias/read:02"v/dense_6/bias/Initializer/zeros:08
w
v/dense_7/kernel:0v/dense_7/kernel/Assignv/dense_7/kernel/read:02-v/dense_7/kernel/Initializer/random_uniform:08
f
v/dense_7/bias:0v/dense_7/bias/Assignv/dense_7/bias/read:02"v/dense_7/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
|
pi/dense/kernel/Adam:0pi/dense/kernel/Adam/Assignpi/dense/kernel/Adam/read:02(pi/dense/kernel/Adam/Initializer/zeros:0
�
pi/dense/kernel/Adam_1:0pi/dense/kernel/Adam_1/Assignpi/dense/kernel/Adam_1/read:02*pi/dense/kernel/Adam_1/Initializer/zeros:0
t
pi/dense/bias/Adam:0pi/dense/bias/Adam/Assignpi/dense/bias/Adam/read:02&pi/dense/bias/Adam/Initializer/zeros:0
|
pi/dense/bias/Adam_1:0pi/dense/bias/Adam_1/Assignpi/dense/bias/Adam_1/read:02(pi/dense/bias/Adam_1/Initializer/zeros:0
�
pi/dense_1/kernel/Adam:0pi/dense_1/kernel/Adam/Assignpi/dense_1/kernel/Adam/read:02*pi/dense_1/kernel/Adam/Initializer/zeros:0
�
pi/dense_1/kernel/Adam_1:0pi/dense_1/kernel/Adam_1/Assignpi/dense_1/kernel/Adam_1/read:02,pi/dense_1/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_1/bias/Adam:0pi/dense_1/bias/Adam/Assignpi/dense_1/bias/Adam/read:02(pi/dense_1/bias/Adam/Initializer/zeros:0
�
pi/dense_1/bias/Adam_1:0pi/dense_1/bias/Adam_1/Assignpi/dense_1/bias/Adam_1/read:02*pi/dense_1/bias/Adam_1/Initializer/zeros:0
�
pi/dense_2/kernel/Adam:0pi/dense_2/kernel/Adam/Assignpi/dense_2/kernel/Adam/read:02*pi/dense_2/kernel/Adam/Initializer/zeros:0
�
pi/dense_2/kernel/Adam_1:0pi/dense_2/kernel/Adam_1/Assignpi/dense_2/kernel/Adam_1/read:02,pi/dense_2/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_2/bias/Adam:0pi/dense_2/bias/Adam/Assignpi/dense_2/bias/Adam/read:02(pi/dense_2/bias/Adam/Initializer/zeros:0
�
pi/dense_2/bias/Adam_1:0pi/dense_2/bias/Adam_1/Assignpi/dense_2/bias/Adam_1/read:02*pi/dense_2/bias/Adam_1/Initializer/zeros:0
�
pi/dense_3/kernel/Adam:0pi/dense_3/kernel/Adam/Assignpi/dense_3/kernel/Adam/read:02*pi/dense_3/kernel/Adam/Initializer/zeros:0
�
pi/dense_3/kernel/Adam_1:0pi/dense_3/kernel/Adam_1/Assignpi/dense_3/kernel/Adam_1/read:02,pi/dense_3/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_3/bias/Adam:0pi/dense_3/bias/Adam/Assignpi/dense_3/bias/Adam/read:02(pi/dense_3/bias/Adam/Initializer/zeros:0
�
pi/dense_3/bias/Adam_1:0pi/dense_3/bias/Adam_1/Assignpi/dense_3/bias/Adam_1/read:02*pi/dense_3/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
x
v/dense/kernel/Adam:0v/dense/kernel/Adam/Assignv/dense/kernel/Adam/read:02'v/dense/kernel/Adam/Initializer/zeros:0
�
v/dense/kernel/Adam_1:0v/dense/kernel/Adam_1/Assignv/dense/kernel/Adam_1/read:02)v/dense/kernel/Adam_1/Initializer/zeros:0
p
v/dense/bias/Adam:0v/dense/bias/Adam/Assignv/dense/bias/Adam/read:02%v/dense/bias/Adam/Initializer/zeros:0
x
v/dense/bias/Adam_1:0v/dense/bias/Adam_1/Assignv/dense/bias/Adam_1/read:02'v/dense/bias/Adam_1/Initializer/zeros:0
�
v/dense_1/kernel/Adam:0v/dense_1/kernel/Adam/Assignv/dense_1/kernel/Adam/read:02)v/dense_1/kernel/Adam/Initializer/zeros:0
�
v/dense_1/kernel/Adam_1:0v/dense_1/kernel/Adam_1/Assignv/dense_1/kernel/Adam_1/read:02+v/dense_1/kernel/Adam_1/Initializer/zeros:0
x
v/dense_1/bias/Adam:0v/dense_1/bias/Adam/Assignv/dense_1/bias/Adam/read:02'v/dense_1/bias/Adam/Initializer/zeros:0
�
v/dense_1/bias/Adam_1:0v/dense_1/bias/Adam_1/Assignv/dense_1/bias/Adam_1/read:02)v/dense_1/bias/Adam_1/Initializer/zeros:0
�
v/dense_2/kernel/Adam:0v/dense_2/kernel/Adam/Assignv/dense_2/kernel/Adam/read:02)v/dense_2/kernel/Adam/Initializer/zeros:0
�
v/dense_2/kernel/Adam_1:0v/dense_2/kernel/Adam_1/Assignv/dense_2/kernel/Adam_1/read:02+v/dense_2/kernel/Adam_1/Initializer/zeros:0
x
v/dense_2/bias/Adam:0v/dense_2/bias/Adam/Assignv/dense_2/bias/Adam/read:02'v/dense_2/bias/Adam/Initializer/zeros:0
�
v/dense_2/bias/Adam_1:0v/dense_2/bias/Adam_1/Assignv/dense_2/bias/Adam_1/read:02)v/dense_2/bias/Adam_1/Initializer/zeros:0
�
v/dense_3/kernel/Adam:0v/dense_3/kernel/Adam/Assignv/dense_3/kernel/Adam/read:02)v/dense_3/kernel/Adam/Initializer/zeros:0
�
v/dense_3/kernel/Adam_1:0v/dense_3/kernel/Adam_1/Assignv/dense_3/kernel/Adam_1/read:02+v/dense_3/kernel/Adam_1/Initializer/zeros:0
x
v/dense_3/bias/Adam:0v/dense_3/bias/Adam/Assignv/dense_3/bias/Adam/read:02'v/dense_3/bias/Adam/Initializer/zeros:0
�
v/dense_3/bias/Adam_1:0v/dense_3/bias/Adam_1/Assignv/dense_3/bias/Adam_1/read:02)v/dense_3/bias/Adam_1/Initializer/zeros:0
�
v/dense_4/kernel/Adam:0v/dense_4/kernel/Adam/Assignv/dense_4/kernel/Adam/read:02)v/dense_4/kernel/Adam/Initializer/zeros:0
�
v/dense_4/kernel/Adam_1:0v/dense_4/kernel/Adam_1/Assignv/dense_4/kernel/Adam_1/read:02+v/dense_4/kernel/Adam_1/Initializer/zeros:0
x
v/dense_4/bias/Adam:0v/dense_4/bias/Adam/Assignv/dense_4/bias/Adam/read:02'v/dense_4/bias/Adam/Initializer/zeros:0
�
v/dense_4/bias/Adam_1:0v/dense_4/bias/Adam_1/Assignv/dense_4/bias/Adam_1/read:02)v/dense_4/bias/Adam_1/Initializer/zeros:0
�
v/dense_5/kernel/Adam:0v/dense_5/kernel/Adam/Assignv/dense_5/kernel/Adam/read:02)v/dense_5/kernel/Adam/Initializer/zeros:0
�
v/dense_5/kernel/Adam_1:0v/dense_5/kernel/Adam_1/Assignv/dense_5/kernel/Adam_1/read:02+v/dense_5/kernel/Adam_1/Initializer/zeros:0
x
v/dense_5/bias/Adam:0v/dense_5/bias/Adam/Assignv/dense_5/bias/Adam/read:02'v/dense_5/bias/Adam/Initializer/zeros:0
�
v/dense_5/bias/Adam_1:0v/dense_5/bias/Adam_1/Assignv/dense_5/bias/Adam_1/read:02)v/dense_5/bias/Adam_1/Initializer/zeros:0
�
v/dense_6/kernel/Adam:0v/dense_6/kernel/Adam/Assignv/dense_6/kernel/Adam/read:02)v/dense_6/kernel/Adam/Initializer/zeros:0
�
v/dense_6/kernel/Adam_1:0v/dense_6/kernel/Adam_1/Assignv/dense_6/kernel/Adam_1/read:02+v/dense_6/kernel/Adam_1/Initializer/zeros:0
x
v/dense_6/bias/Adam:0v/dense_6/bias/Adam/Assignv/dense_6/bias/Adam/read:02'v/dense_6/bias/Adam/Initializer/zeros:0
�
v/dense_6/bias/Adam_1:0v/dense_6/bias/Adam_1/Assignv/dense_6/bias/Adam_1/read:02)v/dense_6/bias/Adam_1/Initializer/zeros:0
�
v/dense_7/kernel/Adam:0v/dense_7/kernel/Adam/Assignv/dense_7/kernel/Adam/read:02)v/dense_7/kernel/Adam/Initializer/zeros:0
�
v/dense_7/kernel/Adam_1:0v/dense_7/kernel/Adam_1/Assignv/dense_7/kernel/Adam_1/read:02+v/dense_7/kernel/Adam_1/Initializer/zeros:0
x
v/dense_7/bias/Adam:0v/dense_7/bias/Adam/Assignv/dense_7/bias/Adam/read:02'v/dense_7/bias/Adam/Initializer/zeros:0
�
v/dense_7/bias/Adam_1:0v/dense_7/bias/Adam_1/Assignv/dense_7/bias/Adam_1/read:02)v/dense_7/bias/Adam_1/Initializer/zeros:0"�
trainable_variables��
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
{
pi/dense_3/kernel:0pi/dense_3/kernel/Assignpi/dense_3/kernel/read:02.pi/dense_3/kernel/Initializer/random_uniform:08
j
pi/dense_3/bias:0pi/dense_3/bias/Assignpi/dense_3/bias/read:02#pi/dense_3/bias/Initializer/zeros:08
o
v/dense/kernel:0v/dense/kernel/Assignv/dense/kernel/read:02+v/dense/kernel/Initializer/random_uniform:08
^
v/dense/bias:0v/dense/bias/Assignv/dense/bias/read:02 v/dense/bias/Initializer/zeros:08
w
v/dense_1/kernel:0v/dense_1/kernel/Assignv/dense_1/kernel/read:02-v/dense_1/kernel/Initializer/random_uniform:08
f
v/dense_1/bias:0v/dense_1/bias/Assignv/dense_1/bias/read:02"v/dense_1/bias/Initializer/zeros:08
w
v/dense_2/kernel:0v/dense_2/kernel/Assignv/dense_2/kernel/read:02-v/dense_2/kernel/Initializer/random_uniform:08
f
v/dense_2/bias:0v/dense_2/bias/Assignv/dense_2/bias/read:02"v/dense_2/bias/Initializer/zeros:08
w
v/dense_3/kernel:0v/dense_3/kernel/Assignv/dense_3/kernel/read:02-v/dense_3/kernel/Initializer/random_uniform:08
f
v/dense_3/bias:0v/dense_3/bias/Assignv/dense_3/bias/read:02"v/dense_3/bias/Initializer/zeros:08
w
v/dense_4/kernel:0v/dense_4/kernel/Assignv/dense_4/kernel/read:02-v/dense_4/kernel/Initializer/random_uniform:08
f
v/dense_4/bias:0v/dense_4/bias/Assignv/dense_4/bias/read:02"v/dense_4/bias/Initializer/zeros:08
w
v/dense_5/kernel:0v/dense_5/kernel/Assignv/dense_5/kernel/read:02-v/dense_5/kernel/Initializer/random_uniform:08
f
v/dense_5/bias:0v/dense_5/bias/Assignv/dense_5/bias/read:02"v/dense_5/bias/Initializer/zeros:08
w
v/dense_6/kernel:0v/dense_6/kernel/Assignv/dense_6/kernel/read:02-v/dense_6/kernel/Initializer/random_uniform:08
f
v/dense_6/bias:0v/dense_6/bias/Assignv/dense_6/bias/read:02"v/dense_6/bias/Initializer/zeros:08
w
v/dense_7/kernel:0v/dense_7/kernel/Assignv/dense_7/kernel/read:02-v/dense_7/kernel/Initializer/random_uniform:08
f
v/dense_7/bias:0v/dense_7/bias/Assignv/dense_7/bias/read:02"v/dense_7/bias/Initializer/zeros:08"
train_pi

Adam"
train_op

Adam
Adam_1*�
serving_default�
/
mask'
Placeholder_2:0����������
)
ret"
Placeholder_4:0���������
1
logp_old_ph"
Placeholder_5:0���������
'
a"
Placeholder_1:0���������
+
x&
Placeholder:0�����������
)
adv"
Placeholder_3:0���������%
v 
v/Squeeze_1:0���������(
logp_pi

pi/Sum_1:0���������
	approx_kl
Mean_2:0 
clipfrac
Mean_4:0 

approx_ent
Mean_3:0 
v_loss
Mean_1:0 '
out 
pi/add:0����������)
clipped
LogicalOr:0
���������#
logp
pi/Sum:0���������
pi_loss
Neg:0 '
pi!
pi/Squeeze_1:0	���������tensorflow/serving/predict