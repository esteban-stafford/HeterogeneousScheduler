��
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
Ttype"serve*1.14.02v1.14.0-rc1-22-gaf24dc91b5��
p
PlaceholderPlaceholder*
shape:����������*(
_output_shapes
:����������*
dtype0
r
Placeholder_1Placeholder*
shape:����������*(
_output_shapes
:����������*
dtype0
r
Placeholder_2Placeholder*
dtype0*(
_output_shapes
:����������*
shape:����������
h
Placeholder_3Placeholder*
shape:���������*#
_output_shapes
:���������*
dtype0
h
Placeholder_4Placeholder*
shape:���������*#
_output_shapes
:���������*
dtype0
r
Placeholder_5Placeholder*
shape:����������*(
_output_shapes
:����������*
dtype0
p
Placeholder_6Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
h
Placeholder_7Placeholder*#
_output_shapes
:���������*
shape:���������*
dtype0
h
Placeholder_8Placeholder*#
_output_shapes
:���������*
shape:���������*
dtype0
h
Placeholder_9Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
i
Placeholder_10Placeholder*#
_output_shapes
:���������*
shape:���������*
dtype0
g
pi_j/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"�����      
}
pi_j/ReshapeReshapePlaceholderpi_j/Reshape/shape*
Tshape0*,
_output_shapes
:����������*
T0
�
2pi_j/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"       *$
_class
loc:@pi_j/dense/kernel
�
0pi_j/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *�Ѿ*
_output_shapes
: *
dtype0*$
_class
loc:@pi_j/dense/kernel
�
0pi_j/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *��>*
dtype0*$
_class
loc:@pi_j/dense/kernel
�
:pi_j/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi_j/dense/kernel/Initializer/random_uniform/shape*
_output_shapes

: *
seed2*
dtype0*$
_class
loc:@pi_j/dense/kernel*
T0*
seed�
�
0pi_j/dense/kernel/Initializer/random_uniform/subSub0pi_j/dense/kernel/Initializer/random_uniform/max0pi_j/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *$
_class
loc:@pi_j/dense/kernel*
T0
�
0pi_j/dense/kernel/Initializer/random_uniform/mulMul:pi_j/dense/kernel/Initializer/random_uniform/RandomUniform0pi_j/dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel
�
,pi_j/dense/kernel/Initializer/random_uniformAdd0pi_j/dense/kernel/Initializer/random_uniform/mul0pi_j/dense/kernel/Initializer/random_uniform/min*$
_class
loc:@pi_j/dense/kernel*
T0*
_output_shapes

: 
�
pi_j/dense/kernel
VariableV2*
	container *
dtype0*
shape
: *
_output_shapes

: *
shared_name *$
_class
loc:@pi_j/dense/kernel
�
pi_j/dense/kernel/AssignAssignpi_j/dense/kernel,pi_j/dense/kernel/Initializer/random_uniform*
_output_shapes

: *
validate_shape(*
use_locking(*$
_class
loc:@pi_j/dense/kernel*
T0
�
pi_j/dense/kernel/readIdentitypi_j/dense/kernel*
T0*
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel
�
!pi_j/dense/bias/Initializer/zerosConst*
valueB *    *
dtype0*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: 
�
pi_j/dense/bias
VariableV2*
shape: *
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
	container *
dtype0*
shared_name 
�
pi_j/dense/bias/AssignAssignpi_j/dense/bias!pi_j/dense/bias/Initializer/zeros*
T0*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
z
pi_j/dense/bias/readIdentitypi_j/dense/bias*
T0*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias
c
pi_j/dense/Tensordot/axesConst*
valueB:*
_output_shapes
:*
dtype0
j
pi_j/dense/Tensordot/freeConst*
_output_shapes
:*
valueB"       *
dtype0
f
pi_j/dense/Tensordot/ShapeShapepi_j/Reshape*
out_type0*
T0*
_output_shapes
:
d
"pi_j/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
pi_j/dense/Tensordot/GatherV2GatherV2pi_j/dense/Tensordot/Shapepi_j/dense/Tensordot/free"pi_j/dense/Tensordot/GatherV2/axis*

batch_dims *
Tindices0*
Tparams0*
Taxis0*
_output_shapes
:
f
$pi_j/dense/Tensordot/GatherV2_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
pi_j/dense/Tensordot/GatherV2_1GatherV2pi_j/dense/Tensordot/Shapepi_j/dense/Tensordot/axes$pi_j/dense/Tensordot/GatherV2_1/axis*
Tparams0*
Taxis0*

batch_dims *
Tindices0*
_output_shapes
:
d
pi_j/dense/Tensordot/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
pi_j/dense/Tensordot/ProdProdpi_j/dense/Tensordot/GatherV2pi_j/dense/Tensordot/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
f
pi_j/dense/Tensordot/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
pi_j/dense/Tensordot/Prod_1Prodpi_j/dense/Tensordot/GatherV2_1pi_j/dense/Tensordot/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
b
 pi_j/dense/Tensordot/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
pi_j/dense/Tensordot/concatConcatV2pi_j/dense/Tensordot/freepi_j/dense/Tensordot/axes pi_j/dense/Tensordot/concat/axis*

Tidx0*
N*
T0*
_output_shapes
:
�
pi_j/dense/Tensordot/stackPackpi_j/dense/Tensordot/Prodpi_j/dense/Tensordot/Prod_1*
N*
_output_shapes
:*
T0*

axis 
�
pi_j/dense/Tensordot/transpose	Transposepi_j/Reshapepi_j/dense/Tensordot/concat*
Tperm0*
T0*,
_output_shapes
:����������
�
pi_j/dense/Tensordot/ReshapeReshapepi_j/dense/Tensordot/transposepi_j/dense/Tensordot/stack*0
_output_shapes
:������������������*
T0*
Tshape0
v
%pi_j/dense/Tensordot/transpose_1/permConst*
valueB"       *
_output_shapes
:*
dtype0
�
 pi_j/dense/Tensordot/transpose_1	Transposepi_j/dense/kernel/read%pi_j/dense/Tensordot/transpose_1/perm*
_output_shapes

: *
T0*
Tperm0
u
$pi_j/dense/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       
�
pi_j/dense/Tensordot/Reshape_1Reshape pi_j/dense/Tensordot/transpose_1$pi_j/dense/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

: 
�
pi_j/dense/Tensordot/MatMulMatMulpi_j/dense/Tensordot/Reshapepi_j/dense/Tensordot/Reshape_1*
T0*'
_output_shapes
:��������� *
transpose_a( *
transpose_b( 
f
pi_j/dense/Tensordot/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
d
"pi_j/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
pi_j/dense/Tensordot/concat_1ConcatV2pi_j/dense/Tensordot/GatherV2pi_j/dense/Tensordot/Const_2"pi_j/dense/Tensordot/concat_1/axis*
T0*
_output_shapes
:*
N*

Tidx0
�
pi_j/dense/TensordotReshapepi_j/dense/Tensordot/MatMulpi_j/dense/Tensordot/concat_1*
T0*,
_output_shapes
:���������� *
Tshape0
�
pi_j/dense/BiasAddBiasAddpi_j/dense/Tensordotpi_j/dense/bias/read*
data_formatNHWC*
T0*,
_output_shapes
:���������� 
b
pi_j/dense/ReluRelupi_j/dense/BiasAdd*
T0*,
_output_shapes
:���������� 
�
4pi_j/dense_1/kernel/Initializer/random_uniform/shapeConst*&
_class
loc:@pi_j/dense_1/kernel*
dtype0*
_output_shapes
:*
valueB"       
�
2pi_j/dense_1/kernel/Initializer/random_uniform/minConst*&
_class
loc:@pi_j/dense_1/kernel*
valueB
 *���*
dtype0*
_output_shapes
: 
�
2pi_j/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *��>*
dtype0*
_output_shapes
: *&
_class
loc:@pi_j/dense_1/kernel
�
<pi_j/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4pi_j/dense_1/kernel/Initializer/random_uniform/shape*
seed�*
seed29*
dtype0*
_output_shapes

: *
T0*&
_class
loc:@pi_j/dense_1/kernel
�
2pi_j/dense_1/kernel/Initializer/random_uniform/subSub2pi_j/dense_1/kernel/Initializer/random_uniform/max2pi_j/dense_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *&
_class
loc:@pi_j/dense_1/kernel
�
2pi_j/dense_1/kernel/Initializer/random_uniform/mulMul<pi_j/dense_1/kernel/Initializer/random_uniform/RandomUniform2pi_j/dense_1/kernel/Initializer/random_uniform/sub*&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: *
T0
�
.pi_j/dense_1/kernel/Initializer/random_uniformAdd2pi_j/dense_1/kernel/Initializer/random_uniform/mul2pi_j/dense_1/kernel/Initializer/random_uniform/min*&
_class
loc:@pi_j/dense_1/kernel*
T0*
_output_shapes

: 
�
pi_j/dense_1/kernel
VariableV2*
shape
: *
shared_name *&
_class
loc:@pi_j/dense_1/kernel*
	container *
_output_shapes

: *
dtype0
�
pi_j/dense_1/kernel/AssignAssignpi_j/dense_1/kernel.pi_j/dense_1/kernel/Initializer/random_uniform*
_output_shapes

: *
T0*&
_class
loc:@pi_j/dense_1/kernel*
validate_shape(*
use_locking(
�
pi_j/dense_1/kernel/readIdentitypi_j/dense_1/kernel*
T0*&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: 
�
#pi_j/dense_1/bias/Initializer/zerosConst*$
_class
loc:@pi_j/dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
�
pi_j/dense_1/bias
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes
:*$
_class
loc:@pi_j/dense_1/bias*
shape:
�
pi_j/dense_1/bias/AssignAssignpi_j/dense_1/bias#pi_j/dense_1/bias/Initializer/zeros*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:
�
pi_j/dense_1/bias/readIdentitypi_j/dense_1/bias*$
_class
loc:@pi_j/dense_1/bias*
T0*
_output_shapes
:
e
pi_j/dense_1/Tensordot/axesConst*
dtype0*
valueB:*
_output_shapes
:
l
pi_j/dense_1/Tensordot/freeConst*
_output_shapes
:*
valueB"       *
dtype0
k
pi_j/dense_1/Tensordot/ShapeShapepi_j/dense/Relu*
T0*
_output_shapes
:*
out_type0
f
$pi_j/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
pi_j/dense_1/Tensordot/GatherV2GatherV2pi_j/dense_1/Tensordot/Shapepi_j/dense_1/Tensordot/free$pi_j/dense_1/Tensordot/GatherV2/axis*
_output_shapes
:*
Tparams0*
Tindices0*

batch_dims *
Taxis0
h
&pi_j/dense_1/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
!pi_j/dense_1/Tensordot/GatherV2_1GatherV2pi_j/dense_1/Tensordot/Shapepi_j/dense_1/Tensordot/axes&pi_j/dense_1/Tensordot/GatherV2_1/axis*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims 
f
pi_j/dense_1/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
pi_j/dense_1/Tensordot/ProdProdpi_j/dense_1/Tensordot/GatherV2pi_j/dense_1/Tensordot/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
h
pi_j/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
pi_j/dense_1/Tensordot/Prod_1Prod!pi_j/dense_1/Tensordot/GatherV2_1pi_j/dense_1/Tensordot/Const_1*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
d
"pi_j/dense_1/Tensordot/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
pi_j/dense_1/Tensordot/concatConcatV2pi_j/dense_1/Tensordot/freepi_j/dense_1/Tensordot/axes"pi_j/dense_1/Tensordot/concat/axis*
T0*

Tidx0*
N*
_output_shapes
:
�
pi_j/dense_1/Tensordot/stackPackpi_j/dense_1/Tensordot/Prodpi_j/dense_1/Tensordot/Prod_1*

axis *
T0*
_output_shapes
:*
N
�
 pi_j/dense_1/Tensordot/transpose	Transposepi_j/dense/Relupi_j/dense_1/Tensordot/concat*,
_output_shapes
:���������� *
Tperm0*
T0
�
pi_j/dense_1/Tensordot/ReshapeReshape pi_j/dense_1/Tensordot/transposepi_j/dense_1/Tensordot/stack*
Tshape0*0
_output_shapes
:������������������*
T0
x
'pi_j/dense_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
�
"pi_j/dense_1/Tensordot/transpose_1	Transposepi_j/dense_1/kernel/read'pi_j/dense_1/Tensordot/transpose_1/perm*
T0*
_output_shapes

: *
Tperm0
w
&pi_j/dense_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       
�
 pi_j/dense_1/Tensordot/Reshape_1Reshape"pi_j/dense_1/Tensordot/transpose_1&pi_j/dense_1/Tensordot/Reshape_1/shape*
Tshape0*
T0*
_output_shapes

: 
�
pi_j/dense_1/Tensordot/MatMulMatMulpi_j/dense_1/Tensordot/Reshape pi_j/dense_1/Tensordot/Reshape_1*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
h
pi_j/dense_1/Tensordot/Const_2Const*
valueB:*
_output_shapes
:*
dtype0
f
$pi_j/dense_1/Tensordot/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
pi_j/dense_1/Tensordot/concat_1ConcatV2pi_j/dense_1/Tensordot/GatherV2pi_j/dense_1/Tensordot/Const_2$pi_j/dense_1/Tensordot/concat_1/axis*
N*
_output_shapes
:*
T0*

Tidx0
�
pi_j/dense_1/TensordotReshapepi_j/dense_1/Tensordot/MatMulpi_j/dense_1/Tensordot/concat_1*,
_output_shapes
:����������*
T0*
Tshape0
�
pi_j/dense_1/BiasAddBiasAddpi_j/dense_1/Tensordotpi_j/dense_1/bias/read*
data_formatNHWC*
T0*,
_output_shapes
:����������
f
pi_j/dense_1/ReluRelupi_j/dense_1/BiasAdd*,
_output_shapes
:����������*
T0
�
4pi_j/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*&
_class
loc:@pi_j/dense_2/kernel
�
2pi_j/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *   �*
_output_shapes
: *
dtype0*&
_class
loc:@pi_j/dense_2/kernel
�
2pi_j/dense_2/kernel/Initializer/random_uniform/maxConst*&
_class
loc:@pi_j/dense_2/kernel*
valueB
 *   ?*
_output_shapes
: *
dtype0
�
<pi_j/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4pi_j/dense_2/kernel/Initializer/random_uniform/shape*&
_class
loc:@pi_j/dense_2/kernel*
dtype0*
seed2b*
_output_shapes

:*
T0*
seed�
�
2pi_j/dense_2/kernel/Initializer/random_uniform/subSub2pi_j/dense_2/kernel/Initializer/random_uniform/max2pi_j/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*&
_class
loc:@pi_j/dense_2/kernel
�
2pi_j/dense_2/kernel/Initializer/random_uniform/mulMul<pi_j/dense_2/kernel/Initializer/random_uniform/RandomUniform2pi_j/dense_2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@pi_j/dense_2/kernel*
_output_shapes

:
�
.pi_j/dense_2/kernel/Initializer/random_uniformAdd2pi_j/dense_2/kernel/Initializer/random_uniform/mul2pi_j/dense_2/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:*&
_class
loc:@pi_j/dense_2/kernel
�
pi_j/dense_2/kernel
VariableV2*&
_class
loc:@pi_j/dense_2/kernel*
shape
:*
dtype0*
	container *
_output_shapes

:*
shared_name 
�
pi_j/dense_2/kernel/AssignAssignpi_j/dense_2/kernel.pi_j/dense_2/kernel/Initializer/random_uniform*
T0*
use_locking(*&
_class
loc:@pi_j/dense_2/kernel*
_output_shapes

:*
validate_shape(
�
pi_j/dense_2/kernel/readIdentitypi_j/dense_2/kernel*
_output_shapes

:*
T0*&
_class
loc:@pi_j/dense_2/kernel
�
#pi_j/dense_2/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*$
_class
loc:@pi_j/dense_2/bias
�
pi_j/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
shape:*
	container *$
_class
loc:@pi_j/dense_2/bias
�
pi_j/dense_2/bias/AssignAssignpi_j/dense_2/bias#pi_j/dense_2/bias/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_j/dense_2/bias
�
pi_j/dense_2/bias/readIdentitypi_j/dense_2/bias*
T0*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:
e
pi_j/dense_2/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
l
pi_j/dense_2/Tensordot/freeConst*
dtype0*
_output_shapes
:*
valueB"       
m
pi_j/dense_2/Tensordot/ShapeShapepi_j/dense_1/Relu*
T0*
out_type0*
_output_shapes
:
f
$pi_j/dense_2/Tensordot/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
pi_j/dense_2/Tensordot/GatherV2GatherV2pi_j/dense_2/Tensordot/Shapepi_j/dense_2/Tensordot/free$pi_j/dense_2/Tensordot/GatherV2/axis*
_output_shapes
:*
Tindices0*
Tparams0*
Taxis0*

batch_dims 
h
&pi_j/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
!pi_j/dense_2/Tensordot/GatherV2_1GatherV2pi_j/dense_2/Tensordot/Shapepi_j/dense_2/Tensordot/axes&pi_j/dense_2/Tensordot/GatherV2_1/axis*
Tindices0*
Tparams0*
_output_shapes
:*

batch_dims *
Taxis0
f
pi_j/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
�
pi_j/dense_2/Tensordot/ProdProdpi_j/dense_2/Tensordot/GatherV2pi_j/dense_2/Tensordot/Const*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
h
pi_j/dense_2/Tensordot/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
pi_j/dense_2/Tensordot/Prod_1Prod!pi_j/dense_2/Tensordot/GatherV2_1pi_j/dense_2/Tensordot/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
d
"pi_j/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
pi_j/dense_2/Tensordot/concatConcatV2pi_j/dense_2/Tensordot/freepi_j/dense_2/Tensordot/axes"pi_j/dense_2/Tensordot/concat/axis*
N*
_output_shapes
:*
T0*

Tidx0
�
pi_j/dense_2/Tensordot/stackPackpi_j/dense_2/Tensordot/Prodpi_j/dense_2/Tensordot/Prod_1*
T0*
N*
_output_shapes
:*

axis 
�
 pi_j/dense_2/Tensordot/transpose	Transposepi_j/dense_1/Relupi_j/dense_2/Tensordot/concat*,
_output_shapes
:����������*
T0*
Tperm0
�
pi_j/dense_2/Tensordot/ReshapeReshape pi_j/dense_2/Tensordot/transposepi_j/dense_2/Tensordot/stack*
T0*0
_output_shapes
:������������������*
Tshape0
x
'pi_j/dense_2/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
�
"pi_j/dense_2/Tensordot/transpose_1	Transposepi_j/dense_2/kernel/read'pi_j/dense_2/Tensordot/transpose_1/perm*
Tperm0*
_output_shapes

:*
T0
w
&pi_j/dense_2/Tensordot/Reshape_1/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
 pi_j/dense_2/Tensordot/Reshape_1Reshape"pi_j/dense_2/Tensordot/transpose_1&pi_j/dense_2/Tensordot/Reshape_1/shape*
Tshape0*
T0*
_output_shapes

:
�
pi_j/dense_2/Tensordot/MatMulMatMulpi_j/dense_2/Tensordot/Reshape pi_j/dense_2/Tensordot/Reshape_1*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:���������
h
pi_j/dense_2/Tensordot/Const_2Const*
dtype0*
valueB:*
_output_shapes
:
f
$pi_j/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
pi_j/dense_2/Tensordot/concat_1ConcatV2pi_j/dense_2/Tensordot/GatherV2pi_j/dense_2/Tensordot/Const_2$pi_j/dense_2/Tensordot/concat_1/axis*
_output_shapes
:*

Tidx0*
T0*
N
�
pi_j/dense_2/TensordotReshapepi_j/dense_2/Tensordot/MatMulpi_j/dense_2/Tensordot/concat_1*
T0*,
_output_shapes
:����������*
Tshape0
�
pi_j/dense_2/BiasAddBiasAddpi_j/dense_2/Tensordotpi_j/dense_2/bias/read*
data_formatNHWC*,
_output_shapes
:����������*
T0
f
pi_j/dense_2/ReluRelupi_j/dense_2/BiasAdd*
T0*,
_output_shapes
:����������
�
4pi_j/dense_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *&
_class
loc:@pi_j/dense_3/kernel*
dtype0
�
2pi_j/dense_3/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*&
_class
loc:@pi_j/dense_3/kernel*
valueB
 *�Q�
�
2pi_j/dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*&
_class
loc:@pi_j/dense_3/kernel*
valueB
 *�Q?*
_output_shapes
: 
�
<pi_j/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform4pi_j/dense_3/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@pi_j/dense_3/kernel*
seed�*
dtype0*
seed2�*
_output_shapes

:
�
2pi_j/dense_3/kernel/Initializer/random_uniform/subSub2pi_j/dense_3/kernel/Initializer/random_uniform/max2pi_j/dense_3/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *&
_class
loc:@pi_j/dense_3/kernel
�
2pi_j/dense_3/kernel/Initializer/random_uniform/mulMul<pi_j/dense_3/kernel/Initializer/random_uniform/RandomUniform2pi_j/dense_3/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:
�
.pi_j/dense_3/kernel/Initializer/random_uniformAdd2pi_j/dense_3/kernel/Initializer/random_uniform/mul2pi_j/dense_3/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:
�
pi_j/dense_3/kernel
VariableV2*&
_class
loc:@pi_j/dense_3/kernel*
	container *
dtype0*
shared_name *
shape
:*
_output_shapes

:
�
pi_j/dense_3/kernel/AssignAssignpi_j/dense_3/kernel.pi_j/dense_3/kernel/Initializer/random_uniform*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel*
validate_shape(*
T0*
use_locking(
�
pi_j/dense_3/kernel/readIdentitypi_j/dense_3/kernel*&
_class
loc:@pi_j/dense_3/kernel*
T0*
_output_shapes

:
�
#pi_j/dense_3/bias/Initializer/zerosConst*
valueB*    *
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias*
dtype0
�
pi_j/dense_3/bias
VariableV2*$
_class
loc:@pi_j/dense_3/bias*
dtype0*
shape:*
_output_shapes
:*
	container *
shared_name 
�
pi_j/dense_3/bias/AssignAssignpi_j/dense_3/bias#pi_j/dense_3/bias/Initializer/zeros*
use_locking(*$
_class
loc:@pi_j/dense_3/bias*
_output_shapes
:*
T0*
validate_shape(
�
pi_j/dense_3/bias/readIdentitypi_j/dense_3/bias*
T0*$
_class
loc:@pi_j/dense_3/bias*
_output_shapes
:
e
pi_j/dense_3/Tensordot/axesConst*
valueB:*
_output_shapes
:*
dtype0
l
pi_j/dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
m
pi_j/dense_3/Tensordot/ShapeShapepi_j/dense_2/Relu*
_output_shapes
:*
T0*
out_type0
f
$pi_j/dense_3/Tensordot/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
pi_j/dense_3/Tensordot/GatherV2GatherV2pi_j/dense_3/Tensordot/Shapepi_j/dense_3/Tensordot/free$pi_j/dense_3/Tensordot/GatherV2/axis*
Taxis0*
_output_shapes
:*
Tparams0*
Tindices0*

batch_dims 
h
&pi_j/dense_3/Tensordot/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!pi_j/dense_3/Tensordot/GatherV2_1GatherV2pi_j/dense_3/Tensordot/Shapepi_j/dense_3/Tensordot/axes&pi_j/dense_3/Tensordot/GatherV2_1/axis*
_output_shapes
:*
Tparams0*

batch_dims *
Tindices0*
Taxis0
f
pi_j/dense_3/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
pi_j/dense_3/Tensordot/ProdProdpi_j/dense_3/Tensordot/GatherV2pi_j/dense_3/Tensordot/Const*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
h
pi_j/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
�
pi_j/dense_3/Tensordot/Prod_1Prod!pi_j/dense_3/Tensordot/GatherV2_1pi_j/dense_3/Tensordot/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
d
"pi_j/dense_3/Tensordot/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
pi_j/dense_3/Tensordot/concatConcatV2pi_j/dense_3/Tensordot/freepi_j/dense_3/Tensordot/axes"pi_j/dense_3/Tensordot/concat/axis*
T0*
_output_shapes
:*
N*

Tidx0
�
pi_j/dense_3/Tensordot/stackPackpi_j/dense_3/Tensordot/Prodpi_j/dense_3/Tensordot/Prod_1*
_output_shapes
:*
N*

axis *
T0
�
 pi_j/dense_3/Tensordot/transpose	Transposepi_j/dense_2/Relupi_j/dense_3/Tensordot/concat*,
_output_shapes
:����������*
Tperm0*
T0
�
pi_j/dense_3/Tensordot/ReshapeReshape pi_j/dense_3/Tensordot/transposepi_j/dense_3/Tensordot/stack*
Tshape0*
T0*0
_output_shapes
:������������������
x
'pi_j/dense_3/Tensordot/transpose_1/permConst*
valueB"       *
_output_shapes
:*
dtype0
�
"pi_j/dense_3/Tensordot/transpose_1	Transposepi_j/dense_3/kernel/read'pi_j/dense_3/Tensordot/transpose_1/perm*
T0*
Tperm0*
_output_shapes

:
w
&pi_j/dense_3/Tensordot/Reshape_1/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
�
 pi_j/dense_3/Tensordot/Reshape_1Reshape"pi_j/dense_3/Tensordot/transpose_1&pi_j/dense_3/Tensordot/Reshape_1/shape*
Tshape0*
T0*
_output_shapes

:
�
pi_j/dense_3/Tensordot/MatMulMatMulpi_j/dense_3/Tensordot/Reshape pi_j/dense_3/Tensordot/Reshape_1*
transpose_a( *'
_output_shapes
:���������*
T0*
transpose_b( 
h
pi_j/dense_3/Tensordot/Const_2Const*
dtype0*
_output_shapes
:*
valueB:
f
$pi_j/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
pi_j/dense_3/Tensordot/concat_1ConcatV2pi_j/dense_3/Tensordot/GatherV2pi_j/dense_3/Tensordot/Const_2$pi_j/dense_3/Tensordot/concat_1/axis*

Tidx0*
_output_shapes
:*
N*
T0
�
pi_j/dense_3/TensordotReshapepi_j/dense_3/Tensordot/MatMulpi_j/dense_3/Tensordot/concat_1*
Tshape0*
T0*,
_output_shapes
:����������
�
pi_j/dense_3/BiasAddBiasAddpi_j/dense_3/Tensordotpi_j/dense_3/bias/read*
data_formatNHWC*
T0*,
_output_shapes
:����������
�
pi_j/SqueezeSqueezepi_j/dense_3/BiasAdd*
T0*
squeeze_dims

���������*(
_output_shapes
:����������
O

pi_j/sub/yConst*
_output_shapes
: *
valueB
 *  �?*
dtype0
]
pi_j/subSubPlaceholder_5
pi_j/sub/y*
T0*(
_output_shapes
:����������
O

pi_j/mul/yConst*
dtype0*
_output_shapes
: *
valueB
 * $tI
X
pi_j/mulMulpi_j/sub
pi_j/mul/y*
T0*(
_output_shapes
:����������
Z
pi_j/addAddpi_j/Squeezepi_j/mul*(
_output_shapes
:����������*
T0
Z
pi_j/LogSoftmax
LogSoftmaxpi_j/add*(
_output_shapes
:����������*
T0
j
(pi_j/multinomial/Multinomial/num_samplesConst*
dtype0*
_output_shapes
: *
value	B :
�
pi_j/multinomial/MultinomialMultinomialpi_j/add(pi_j/multinomial/Multinomial/num_samples*
output_dtype0	*'
_output_shapes
:���������*
T0*
seed2�*
seed�
|
pi_j/Squeeze_1Squeezepi_j/multinomial/Multinomial*#
_output_shapes
:���������*
squeeze_dims
*
T0	
Z
pi_j/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?
[
pi_j/one_hot/off_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
U
pi_j/one_hot/depthConst*
value
B :�*
dtype0*
_output_shapes
: 
�
pi_j/one_hotOneHotPlaceholder_3pi_j/one_hot/depthpi_j/one_hot/on_valuepi_j/one_hot/off_value*
TI0*
T0*(
_output_shapes
:����������*
axis���������
c

pi_j/mul_1Mulpi_j/one_hotpi_j/LogSoftmax*(
_output_shapes
:����������*
T0
\
pi_j/Sum/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
�
pi_j/SumSum
pi_j/mul_1pi_j/Sum/reduction_indices*
	keep_dims( *
T0*#
_output_shapes
:���������*

Tidx0
\
pi_j/one_hot_1/on_valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
]
pi_j/one_hot_1/off_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
W
pi_j/one_hot_1/depthConst*
dtype0*
_output_shapes
: *
value
B :�
�
pi_j/one_hot_1OneHotpi_j/Squeeze_1pi_j/one_hot_1/depthpi_j/one_hot_1/on_valuepi_j/one_hot_1/off_value*
TI0	*
T0*
axis���������*(
_output_shapes
:����������
e

pi_j/mul_2Mulpi_j/one_hot_1pi_j/LogSoftmax*
T0*(
_output_shapes
:����������
^
pi_j/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�

pi_j/Sum_1Sum
pi_j/mul_2pi_j/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
`
v/Reshape/shapeConst*
valueB"����<  *
_output_shapes
:*
dtype0
u
	v/ReshapeReshapePlaceholder_2v/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:����������
�
/v/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@v/dense/kernel*
valueB"<      *
_output_shapes
:
�
-v/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *!
_class
loc:@v/dense/kernel*
valueB
 *�̽
�
-v/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *��=*
_output_shapes
: *!
_class
loc:@v/dense/kernel*
dtype0
�
7v/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform/v/dense/kernel/Initializer/random_uniform/shape*!
_class
loc:@v/dense/kernel*
seed2�*
seed�*
T0*
dtype0*
_output_shapes
:	� 
�
-v/dense/kernel/Initializer/random_uniform/subSub-v/dense/kernel/Initializer/random_uniform/max-v/dense/kernel/Initializer/random_uniform/min*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes
: 
�
-v/dense/kernel/Initializer/random_uniform/mulMul7v/dense/kernel/Initializer/random_uniform/RandomUniform-v/dense/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	� 
�
)v/dense/kernel/Initializer/random_uniformAdd-v/dense/kernel/Initializer/random_uniform/mul-v/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	� *!
_class
loc:@v/dense/kernel*
T0
�
v/dense/kernel
VariableV2*
shared_name *!
_class
loc:@v/dense/kernel*
_output_shapes
:	� *
	container *
dtype0*
shape:	� 
�
v/dense/kernel/AssignAssignv/dense/kernel)v/dense/kernel/Initializer/random_uniform*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	� 
|
v/dense/kernel/readIdentityv/dense/kernel*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes
:	� 
�
v/dense/bias/Initializer/zerosConst*
_output_shapes
: *
valueB *    *
_class
loc:@v/dense/bias*
dtype0
�
v/dense/bias
VariableV2*
shared_name *
_output_shapes
: *
shape: *
dtype0*
	container *
_class
loc:@v/dense/bias
�
v/dense/bias/AssignAssignv/dense/biasv/dense/bias/Initializer/zeros*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
q
v/dense/bias/readIdentityv/dense/bias*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: 
�
v/dense/MatMulMatMul	v/Reshapev/dense/kernel/read*'
_output_shapes
:��������� *
transpose_a( *
transpose_b( *
T0
�
v/dense/BiasAddBiasAddv/dense/MatMulv/dense/bias/read*'
_output_shapes
:��������� *
data_formatNHWC*
T0
W
v/dense/ReluReluv/dense/BiasAdd*
T0*'
_output_shapes
:��������� 
�
1v/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"       *#
_class
loc:@v/dense_1/kernel*
dtype0
�
/v/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *���*#
_class
loc:@v/dense_1/kernel
�
/v/dense_1/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@v/dense_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *��>
�
9v/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_1/kernel/Initializer/random_uniform/shape*#
_class
loc:@v/dense_1/kernel*
dtype0*
seed2�*
seed�*
T0*
_output_shapes

: 
�
/v/dense_1/kernel/Initializer/random_uniform/subSub/v/dense_1/kernel/Initializer/random_uniform/max/v/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *#
_class
loc:@v/dense_1/kernel*
T0
�
/v/dense_1/kernel/Initializer/random_uniform/mulMul9v/dense_1/kernel/Initializer/random_uniform/RandomUniform/v/dense_1/kernel/Initializer/random_uniform/sub*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
T0
�
+v/dense_1/kernel/Initializer/random_uniformAdd/v/dense_1/kernel/Initializer/random_uniform/mul/v/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

: *
T0*#
_class
loc:@v/dense_1/kernel
�
v/dense_1/kernel
VariableV2*
	container *
shared_name *
dtype0*#
_class
loc:@v/dense_1/kernel*
shape
: *
_output_shapes

: 
�
v/dense_1/kernel/AssignAssignv/dense_1/kernel+v/dense_1/kernel/Initializer/random_uniform*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
use_locking(*
validate_shape(*
T0
�
v/dense_1/kernel/readIdentityv/dense_1/kernel*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0
�
 v/dense_1/bias/Initializer/zerosConst*!
_class
loc:@v/dense_1/bias*
dtype0*
valueB*    *
_output_shapes
:
�
v/dense_1/bias
VariableV2*!
_class
loc:@v/dense_1/bias*
shape:*
_output_shapes
:*
shared_name *
	container *
dtype0
�
v/dense_1/bias/AssignAssignv/dense_1/bias v/dense_1/bias/Initializer/zeros*
T0*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(
w
v/dense_1/bias/readIdentityv/dense_1/bias*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
T0
�
v/dense_1/MatMulMatMulv/dense/Reluv/dense_1/kernel/read*'
_output_shapes
:���������*
T0*
transpose_b( *
transpose_a( 
�
v/dense_1/BiasAddBiasAddv/dense_1/MatMulv/dense_1/bias/read*'
_output_shapes
:���������*
data_formatNHWC*
T0
[
v/dense_1/ReluReluv/dense_1/BiasAdd*
T0*'
_output_shapes
:���������
�
1v/dense_2/kernel/Initializer/random_uniform/shapeConst*#
_class
loc:@v/dense_2/kernel*
valueB"      *
dtype0*
_output_shapes
:
�
/v/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *   �*
_output_shapes
: *#
_class
loc:@v/dense_2/kernel*
dtype0
�
/v/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *   ?*
_output_shapes
: *
dtype0*#
_class
loc:@v/dense_2/kernel
�
9v/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_2/kernel/Initializer/random_uniform/shape*#
_class
loc:@v/dense_2/kernel*
seed�*
_output_shapes

:*
T0*
seed2�*
dtype0
�
/v/dense_2/kernel/Initializer/random_uniform/subSub/v/dense_2/kernel/Initializer/random_uniform/max/v/dense_2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes
: 
�
/v/dense_2/kernel/Initializer/random_uniform/mulMul9v/dense_2/kernel/Initializer/random_uniform/RandomUniform/v/dense_2/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:
�
+v/dense_2/kernel/Initializer/random_uniformAdd/v/dense_2/kernel/Initializer/random_uniform/mul/v/dense_2/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
T0
�
v/dense_2/kernel
VariableV2*#
_class
loc:@v/dense_2/kernel*
shared_name *
dtype0*
shape
:*
_output_shapes

:*
	container 
�
v/dense_2/kernel/AssignAssignv/dense_2/kernel+v/dense_2/kernel/Initializer/random_uniform*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
use_locking(*
T0
�
v/dense_2/kernel/readIdentityv/dense_2/kernel*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:
�
 v/dense_2/bias/Initializer/zerosConst*!
_class
loc:@v/dense_2/bias*
dtype0*
_output_shapes
:*
valueB*    
�
v/dense_2/bias
VariableV2*
dtype0*!
_class
loc:@v/dense_2/bias*
shape:*
_output_shapes
:*
shared_name *
	container 
�
v/dense_2/bias/AssignAssignv/dense_2/bias v/dense_2/bias/Initializer/zeros*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
w
v/dense_2/bias/readIdentityv/dense_2/bias*!
_class
loc:@v/dense_2/bias*
T0*
_output_shapes
:
�
v/dense_2/MatMulMatMulv/dense_1/Reluv/dense_2/kernel/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:���������
�
v/dense_2/BiasAddBiasAddv/dense_2/MatMulv/dense_2/bias/read*'
_output_shapes
:���������*
data_formatNHWC*
T0
[
v/dense_2/ReluReluv/dense_2/BiasAdd*'
_output_shapes
:���������*
T0
�
1v/dense_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@v/dense_3/kernel*
valueB"      
�
/v/dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *#
_class
loc:@v/dense_3/kernel*
valueB
 *�Q�
�
/v/dense_3/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@v/dense_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *�Q?
�
9v/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_3/kernel/Initializer/random_uniform/shape*
seed�*
seed2�*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
T0*
dtype0
�
/v/dense_3/kernel/Initializer/random_uniform/subSub/v/dense_3/kernel/Initializer/random_uniform/max/v/dense_3/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_3/kernel*
_output_shapes
: 
�
/v/dense_3/kernel/Initializer/random_uniform/mulMul9v/dense_3/kernel/Initializer/random_uniform/RandomUniform/v/dense_3/kernel/Initializer/random_uniform/sub*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
T0
�
+v/dense_3/kernel/Initializer/random_uniformAdd/v/dense_3/kernel/Initializer/random_uniform/mul/v/dense_3/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_3/kernel*
T0*
_output_shapes

:
�
v/dense_3/kernel
VariableV2*
shared_name *
shape
:*
dtype0*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
	container 
�
v/dense_3/kernel/AssignAssignv/dense_3/kernel+v/dense_3/kernel/Initializer/random_uniform*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
use_locking(*
validate_shape(*
T0
�
v/dense_3/kernel/readIdentityv/dense_3/kernel*
T0*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:
�
 v/dense_3/bias/Initializer/zerosConst*
dtype0*
valueB*    *!
_class
loc:@v/dense_3/bias*
_output_shapes
:
�
v/dense_3/bias
VariableV2*
	container *
_output_shapes
:*
shared_name *
shape:*
dtype0*!
_class
loc:@v/dense_3/bias
�
v/dense_3/bias/AssignAssignv/dense_3/bias v/dense_3/bias/Initializer/zeros*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_3/bias
w
v/dense_3/bias/readIdentityv/dense_3/bias*!
_class
loc:@v/dense_3/bias*
T0*
_output_shapes
:
�
v/dense_3/MatMulMatMulv/dense_2/Reluv/dense_3/kernel/read*'
_output_shapes
:���������*
transpose_b( *
transpose_a( *
T0
�
v/dense_3/BiasAddBiasAddv/dense_3/MatMulv/dense_3/bias/read*'
_output_shapes
:���������*
data_formatNHWC*
T0
u
	v/SqueezeSqueezev/dense_3/BiasAdd*
T0*#
_output_shapes
:���������*
squeeze_dims

���������
g
pi_n/Reshape/shapeConst*
dtype0*
_output_shapes
:*!
valueB"����      
~
pi_n/ReshapeReshapePlaceholder_1pi_n/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:���������
�
2pi_n/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*$
_class
loc:@pi_n/dense/kernel*
dtype0*
valueB"       
�
0pi_n/dense/kernel/Initializer/random_uniform/minConst*$
_class
loc:@pi_n/dense/kernel*
dtype0*
valueB
 *��Ⱦ*
_output_shapes
: 
�
0pi_n/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@pi_n/dense/kernel*
valueB
 *���>*
dtype0
�
:pi_n/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi_n/dense/kernel/Initializer/random_uniform/shape*
dtype0*$
_class
loc:@pi_n/dense/kernel*
seed�*
_output_shapes

: *
seed2�*
T0
�
0pi_n/dense/kernel/Initializer/random_uniform/subSub0pi_n/dense/kernel/Initializer/random_uniform/max0pi_n/dense/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi_n/dense/kernel*
_output_shapes
: 
�
0pi_n/dense/kernel/Initializer/random_uniform/mulMul:pi_n/dense/kernel/Initializer/random_uniform/RandomUniform0pi_n/dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel
�
,pi_n/dense/kernel/Initializer/random_uniformAdd0pi_n/dense/kernel/Initializer/random_uniform/mul0pi_n/dense/kernel/Initializer/random_uniform/min*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: *
T0
�
pi_n/dense/kernel
VariableV2*
_output_shapes

: *
shared_name *
shape
: *
	container *$
_class
loc:@pi_n/dense/kernel*
dtype0
�
pi_n/dense/kernel/AssignAssignpi_n/dense/kernel,pi_n/dense/kernel/Initializer/random_uniform*$
_class
loc:@pi_n/dense/kernel*
validate_shape(*
T0*
_output_shapes

: *
use_locking(
�
pi_n/dense/kernel/readIdentitypi_n/dense/kernel*
T0*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: 
�
!pi_n/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *"
_class
loc:@pi_n/dense/bias
�
pi_n/dense/bias
VariableV2*
dtype0*
_output_shapes
: *
	container *"
_class
loc:@pi_n/dense/bias*
shared_name *
shape: 
�
pi_n/dense/bias/AssignAssignpi_n/dense/bias!pi_n/dense/bias/Initializer/zeros*"
_class
loc:@pi_n/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(*
T0
z
pi_n/dense/bias/readIdentitypi_n/dense/bias*"
_class
loc:@pi_n/dense/bias*
T0*
_output_shapes
: 
c
pi_n/dense/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
j
pi_n/dense/Tensordot/freeConst*
dtype0*
valueB"       *
_output_shapes
:
f
pi_n/dense/Tensordot/ShapeShapepi_n/Reshape*
T0*
_output_shapes
:*
out_type0
d
"pi_n/dense/Tensordot/GatherV2/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
pi_n/dense/Tensordot/GatherV2GatherV2pi_n/dense/Tensordot/Shapepi_n/dense/Tensordot/free"pi_n/dense/Tensordot/GatherV2/axis*
Taxis0*
_output_shapes
:*
Tparams0*
Tindices0*

batch_dims 
f
$pi_n/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
pi_n/dense/Tensordot/GatherV2_1GatherV2pi_n/dense/Tensordot/Shapepi_n/dense/Tensordot/axes$pi_n/dense/Tensordot/GatherV2_1/axis*
_output_shapes
:*
Tindices0*

batch_dims *
Taxis0*
Tparams0
d
pi_n/dense/Tensordot/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
pi_n/dense/Tensordot/ProdProdpi_n/dense/Tensordot/GatherV2pi_n/dense/Tensordot/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
f
pi_n/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
�
pi_n/dense/Tensordot/Prod_1Prodpi_n/dense/Tensordot/GatherV2_1pi_n/dense/Tensordot/Const_1*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
b
 pi_n/dense/Tensordot/concat/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
pi_n/dense/Tensordot/concatConcatV2pi_n/dense/Tensordot/freepi_n/dense/Tensordot/axes pi_n/dense/Tensordot/concat/axis*
N*
T0*
_output_shapes
:*

Tidx0
�
pi_n/dense/Tensordot/stackPackpi_n/dense/Tensordot/Prodpi_n/dense/Tensordot/Prod_1*
_output_shapes
:*
T0*
N*

axis 
�
pi_n/dense/Tensordot/transpose	Transposepi_n/Reshapepi_n/dense/Tensordot/concat*+
_output_shapes
:���������*
Tperm0*
T0
�
pi_n/dense/Tensordot/ReshapeReshapepi_n/dense/Tensordot/transposepi_n/dense/Tensordot/stack*
Tshape0*0
_output_shapes
:������������������*
T0
v
%pi_n/dense/Tensordot/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
 pi_n/dense/Tensordot/transpose_1	Transposepi_n/dense/kernel/read%pi_n/dense/Tensordot/transpose_1/perm*
T0*
_output_shapes

: *
Tperm0
u
$pi_n/dense/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"       
�
pi_n/dense/Tensordot/Reshape_1Reshape pi_n/dense/Tensordot/transpose_1$pi_n/dense/Tensordot/Reshape_1/shape*
_output_shapes

: *
Tshape0*
T0
�
pi_n/dense/Tensordot/MatMulMatMulpi_n/dense/Tensordot/Reshapepi_n/dense/Tensordot/Reshape_1*
T0*'
_output_shapes
:��������� *
transpose_a( *
transpose_b( 
f
pi_n/dense/Tensordot/Const_2Const*
_output_shapes
:*
valueB: *
dtype0
d
"pi_n/dense/Tensordot/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
pi_n/dense/Tensordot/concat_1ConcatV2pi_n/dense/Tensordot/GatherV2pi_n/dense/Tensordot/Const_2"pi_n/dense/Tensordot/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
pi_n/dense/TensordotReshapepi_n/dense/Tensordot/MatMulpi_n/dense/Tensordot/concat_1*
T0*
Tshape0*+
_output_shapes
:��������� 
�
pi_n/dense/BiasAddBiasAddpi_n/dense/Tensordotpi_n/dense/bias/read*
data_formatNHWC*+
_output_shapes
:��������� *
T0
a
pi_n/dense/ReluRelupi_n/dense/BiasAdd*
T0*+
_output_shapes
:��������� 
�
4pi_n/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*&
_class
loc:@pi_n/dense_1/kernel*
valueB"       
�
2pi_n/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *���*
_output_shapes
: *&
_class
loc:@pi_n/dense_1/kernel*
dtype0
�
2pi_n/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *��>*
dtype0*&
_class
loc:@pi_n/dense_1/kernel
�
<pi_n/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform4pi_n/dense_1/kernel/Initializer/random_uniform/shape*
T0*&
_class
loc:@pi_n/dense_1/kernel*
seed�*
seed2�*
_output_shapes

: *
dtype0
�
2pi_n/dense_1/kernel/Initializer/random_uniform/subSub2pi_n/dense_1/kernel/Initializer/random_uniform/max2pi_n/dense_1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@pi_n/dense_1/kernel*
_output_shapes
: 
�
2pi_n/dense_1/kernel/Initializer/random_uniform/mulMul<pi_n/dense_1/kernel/Initializer/random_uniform/RandomUniform2pi_n/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
T0
�
.pi_n/dense_1/kernel/Initializer/random_uniformAdd2pi_n/dense_1/kernel/Initializer/random_uniform/mul2pi_n/dense_1/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@pi_n/dense_1/kernel*
_output_shapes

: 
�
pi_n/dense_1/kernel
VariableV2*
	container *
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
dtype0*
shared_name *
shape
: 
�
pi_n/dense_1/kernel/AssignAssignpi_n/dense_1/kernel.pi_n/dense_1/kernel/Initializer/random_uniform*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
use_locking(*
validate_shape(*
T0
�
pi_n/dense_1/kernel/readIdentitypi_n/dense_1/kernel*
T0*&
_class
loc:@pi_n/dense_1/kernel*
_output_shapes

: 
�
#pi_n/dense_1/bias/Initializer/zerosConst*
_output_shapes
:*
valueB*    *$
_class
loc:@pi_n/dense_1/bias*
dtype0
�
pi_n/dense_1/bias
VariableV2*
	container *
shape:*$
_class
loc:@pi_n/dense_1/bias*
dtype0*
_output_shapes
:*
shared_name 
�
pi_n/dense_1/bias/AssignAssignpi_n/dense_1/bias#pi_n/dense_1/bias/Initializer/zeros*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_1/bias*
use_locking(*
validate_shape(
�
pi_n/dense_1/bias/readIdentitypi_n/dense_1/bias*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:*
T0
e
pi_n/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
l
pi_n/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
k
pi_n/dense_1/Tensordot/ShapeShapepi_n/dense/Relu*
out_type0*
T0*
_output_shapes
:
f
$pi_n/dense_1/Tensordot/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
pi_n/dense_1/Tensordot/GatherV2GatherV2pi_n/dense_1/Tensordot/Shapepi_n/dense_1/Tensordot/free$pi_n/dense_1/Tensordot/GatherV2/axis*
Tparams0*
Tindices0*

batch_dims *
_output_shapes
:*
Taxis0
h
&pi_n/dense_1/Tensordot/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
!pi_n/dense_1/Tensordot/GatherV2_1GatherV2pi_n/dense_1/Tensordot/Shapepi_n/dense_1/Tensordot/axes&pi_n/dense_1/Tensordot/GatherV2_1/axis*
Tparams0*
Taxis0*

batch_dims *
_output_shapes
:*
Tindices0
f
pi_n/dense_1/Tensordot/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
pi_n/dense_1/Tensordot/ProdProdpi_n/dense_1/Tensordot/GatherV2pi_n/dense_1/Tensordot/Const*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
h
pi_n/dense_1/Tensordot/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
�
pi_n/dense_1/Tensordot/Prod_1Prod!pi_n/dense_1/Tensordot/GatherV2_1pi_n/dense_1/Tensordot/Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
d
"pi_n/dense_1/Tensordot/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
pi_n/dense_1/Tensordot/concatConcatV2pi_n/dense_1/Tensordot/freepi_n/dense_1/Tensordot/axes"pi_n/dense_1/Tensordot/concat/axis*
_output_shapes
:*
N*

Tidx0*
T0
�
pi_n/dense_1/Tensordot/stackPackpi_n/dense_1/Tensordot/Prodpi_n/dense_1/Tensordot/Prod_1*
T0*
N*
_output_shapes
:*

axis 
�
 pi_n/dense_1/Tensordot/transpose	Transposepi_n/dense/Relupi_n/dense_1/Tensordot/concat*
Tperm0*+
_output_shapes
:��������� *
T0
�
pi_n/dense_1/Tensordot/ReshapeReshape pi_n/dense_1/Tensordot/transposepi_n/dense_1/Tensordot/stack*
Tshape0*
T0*0
_output_shapes
:������������������
x
'pi_n/dense_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
�
"pi_n/dense_1/Tensordot/transpose_1	Transposepi_n/dense_1/kernel/read'pi_n/dense_1/Tensordot/transpose_1/perm*
Tperm0*
_output_shapes

: *
T0
w
&pi_n/dense_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
valueB"       *
dtype0
�
 pi_n/dense_1/Tensordot/Reshape_1Reshape"pi_n/dense_1/Tensordot/transpose_1&pi_n/dense_1/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

: 
�
pi_n/dense_1/Tensordot/MatMulMatMulpi_n/dense_1/Tensordot/Reshape pi_n/dense_1/Tensordot/Reshape_1*
transpose_b( *'
_output_shapes
:���������*
transpose_a( *
T0
h
pi_n/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:
f
$pi_n/dense_1/Tensordot/concat_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
pi_n/dense_1/Tensordot/concat_1ConcatV2pi_n/dense_1/Tensordot/GatherV2pi_n/dense_1/Tensordot/Const_2$pi_n/dense_1/Tensordot/concat_1/axis*
N*

Tidx0*
_output_shapes
:*
T0
�
pi_n/dense_1/TensordotReshapepi_n/dense_1/Tensordot/MatMulpi_n/dense_1/Tensordot/concat_1*+
_output_shapes
:���������*
Tshape0*
T0
�
pi_n/dense_1/BiasAddBiasAddpi_n/dense_1/Tensordotpi_n/dense_1/bias/read*+
_output_shapes
:���������*
data_formatNHWC*
T0
e
pi_n/dense_1/ReluRelupi_n/dense_1/BiasAdd*+
_output_shapes
:���������*
T0
�
4pi_n/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"      *&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes
:
�
2pi_n/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *   �*
dtype0*
_output_shapes
: *&
_class
loc:@pi_n/dense_2/kernel
�
2pi_n/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *   ?*&
_class
loc:@pi_n/dense_2/kernel
�
<pi_n/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform4pi_n/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*
seed2�*
seed�*
T0*&
_class
loc:@pi_n/dense_2/kernel
�
2pi_n/dense_2/kernel/Initializer/random_uniform/subSub2pi_n/dense_2/kernel/Initializer/random_uniform/max2pi_n/dense_2/kernel/Initializer/random_uniform/min*&
_class
loc:@pi_n/dense_2/kernel*
T0*
_output_shapes
: 
�
2pi_n/dense_2/kernel/Initializer/random_uniform/mulMul<pi_n/dense_2/kernel/Initializer/random_uniform/RandomUniform2pi_n/dense_2/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:
�
.pi_n/dense_2/kernel/Initializer/random_uniformAdd2pi_n/dense_2/kernel/Initializer/random_uniform/mul2pi_n/dense_2/kernel/Initializer/random_uniform/min*
T0*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:
�
pi_n/dense_2/kernel
VariableV2*&
_class
loc:@pi_n/dense_2/kernel*
shared_name *
	container *
shape
:*
dtype0*
_output_shapes

:
�
pi_n/dense_2/kernel/AssignAssignpi_n/dense_2/kernel.pi_n/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
_output_shapes

:*
T0*&
_class
loc:@pi_n/dense_2/kernel
�
pi_n/dense_2/kernel/readIdentitypi_n/dense_2/kernel*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:*
T0
�
#pi_n/dense_2/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *$
_class
loc:@pi_n/dense_2/bias
�
pi_n/dense_2/bias
VariableV2*
	container *
dtype0*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
shared_name *
shape:
�
pi_n/dense_2/bias/AssignAssignpi_n/dense_2/bias#pi_n/dense_2/bias/Initializer/zeros*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
�
pi_n/dense_2/bias/readIdentitypi_n/dense_2/bias*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_2/bias
e
pi_n/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
l
pi_n/dense_2/Tensordot/freeConst*
_output_shapes
:*
valueB"       *
dtype0
m
pi_n/dense_2/Tensordot/ShapeShapepi_n/dense_1/Relu*
T0*
_output_shapes
:*
out_type0
f
$pi_n/dense_2/Tensordot/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
pi_n/dense_2/Tensordot/GatherV2GatherV2pi_n/dense_2/Tensordot/Shapepi_n/dense_2/Tensordot/free$pi_n/dense_2/Tensordot/GatherV2/axis*
Tindices0*
_output_shapes
:*

batch_dims *
Taxis0*
Tparams0
h
&pi_n/dense_2/Tensordot/GatherV2_1/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
!pi_n/dense_2/Tensordot/GatherV2_1GatherV2pi_n/dense_2/Tensordot/Shapepi_n/dense_2/Tensordot/axes&pi_n/dense_2/Tensordot/GatherV2_1/axis*
Tparams0*

batch_dims *
Tindices0*
Taxis0*
_output_shapes
:
f
pi_n/dense_2/Tensordot/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
pi_n/dense_2/Tensordot/ProdProdpi_n/dense_2/Tensordot/GatherV2pi_n/dense_2/Tensordot/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
h
pi_n/dense_2/Tensordot/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
�
pi_n/dense_2/Tensordot/Prod_1Prod!pi_n/dense_2/Tensordot/GatherV2_1pi_n/dense_2/Tensordot/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
d
"pi_n/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
pi_n/dense_2/Tensordot/concatConcatV2pi_n/dense_2/Tensordot/freepi_n/dense_2/Tensordot/axes"pi_n/dense_2/Tensordot/concat/axis*

Tidx0*
_output_shapes
:*
T0*
N
�
pi_n/dense_2/Tensordot/stackPackpi_n/dense_2/Tensordot/Prodpi_n/dense_2/Tensordot/Prod_1*
N*

axis *
T0*
_output_shapes
:
�
 pi_n/dense_2/Tensordot/transpose	Transposepi_n/dense_1/Relupi_n/dense_2/Tensordot/concat*+
_output_shapes
:���������*
Tperm0*
T0
�
pi_n/dense_2/Tensordot/ReshapeReshape pi_n/dense_2/Tensordot/transposepi_n/dense_2/Tensordot/stack*
Tshape0*0
_output_shapes
:������������������*
T0
x
'pi_n/dense_2/Tensordot/transpose_1/permConst*
dtype0*
_output_shapes
:*
valueB"       
�
"pi_n/dense_2/Tensordot/transpose_1	Transposepi_n/dense_2/kernel/read'pi_n/dense_2/Tensordot/transpose_1/perm*
_output_shapes

:*
Tperm0*
T0
w
&pi_n/dense_2/Tensordot/Reshape_1/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
�
 pi_n/dense_2/Tensordot/Reshape_1Reshape"pi_n/dense_2/Tensordot/transpose_1&pi_n/dense_2/Tensordot/Reshape_1/shape*
_output_shapes

:*
T0*
Tshape0
�
pi_n/dense_2/Tensordot/MatMulMatMulpi_n/dense_2/Tensordot/Reshape pi_n/dense_2/Tensordot/Reshape_1*
T0*
transpose_b( *'
_output_shapes
:���������*
transpose_a( 
h
pi_n/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
valueB:*
dtype0
f
$pi_n/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
�
pi_n/dense_2/Tensordot/concat_1ConcatV2pi_n/dense_2/Tensordot/GatherV2pi_n/dense_2/Tensordot/Const_2$pi_n/dense_2/Tensordot/concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
�
pi_n/dense_2/TensordotReshapepi_n/dense_2/Tensordot/MatMulpi_n/dense_2/Tensordot/concat_1*+
_output_shapes
:���������*
Tshape0*
T0
�
pi_n/dense_2/BiasAddBiasAddpi_n/dense_2/Tensordotpi_n/dense_2/bias/read*
data_formatNHWC*+
_output_shapes
:���������*
T0
e
pi_n/dense_2/ReluRelupi_n/dense_2/BiasAdd*+
_output_shapes
:���������*
T0
�
4pi_n/dense_3/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *&
_class
loc:@pi_n/dense_3/kernel*
dtype0
�
2pi_n/dense_3/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *�Q�*&
_class
loc:@pi_n/dense_3/kernel*
dtype0
�
2pi_n/dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*&
_class
loc:@pi_n/dense_3/kernel*
valueB
 *�Q?*
_output_shapes
: 
�
<pi_n/dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform4pi_n/dense_3/kernel/Initializer/random_uniform/shape*
seed�*
seed2�*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel*
dtype0*
T0
�
2pi_n/dense_3/kernel/Initializer/random_uniform/subSub2pi_n/dense_3/kernel/Initializer/random_uniform/max2pi_n/dense_3/kernel/Initializer/random_uniform/min*&
_class
loc:@pi_n/dense_3/kernel*
T0*
_output_shapes
: 
�
2pi_n/dense_3/kernel/Initializer/random_uniform/mulMul<pi_n/dense_3/kernel/Initializer/random_uniform/RandomUniform2pi_n/dense_3/kernel/Initializer/random_uniform/sub*
T0*&
_class
loc:@pi_n/dense_3/kernel*
_output_shapes

:
�
.pi_n/dense_3/kernel/Initializer/random_uniformAdd2pi_n/dense_3/kernel/Initializer/random_uniform/mul2pi_n/dense_3/kernel/Initializer/random_uniform/min*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel
�
pi_n/dense_3/kernel
VariableV2*&
_class
loc:@pi_n/dense_3/kernel*
shape
:*
shared_name *
	container *
dtype0*
_output_shapes

:
�
pi_n/dense_3/kernel/AssignAssignpi_n/dense_3/kernel.pi_n/dense_3/kernel/Initializer/random_uniform*
T0*
use_locking(*
validate_shape(*&
_class
loc:@pi_n/dense_3/kernel*
_output_shapes

:
�
pi_n/dense_3/kernel/readIdentitypi_n/dense_3/kernel*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel
�
#pi_n/dense_3/bias/Initializer/zerosConst*$
_class
loc:@pi_n/dense_3/bias*
valueB*    *
_output_shapes
:*
dtype0
�
pi_n/dense_3/bias
VariableV2*
_output_shapes
:*
dtype0*$
_class
loc:@pi_n/dense_3/bias*
shape:*
	container *
shared_name 
�
pi_n/dense_3/bias/AssignAssignpi_n/dense_3/bias#pi_n/dense_3/bias/Initializer/zeros*$
_class
loc:@pi_n/dense_3/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
pi_n/dense_3/bias/readIdentitypi_n/dense_3/bias*$
_class
loc:@pi_n/dense_3/bias*
_output_shapes
:*
T0
e
pi_n/dense_3/Tensordot/axesConst*
valueB:*
dtype0*
_output_shapes
:
l
pi_n/dense_3/Tensordot/freeConst*
valueB"       *
dtype0*
_output_shapes
:
m
pi_n/dense_3/Tensordot/ShapeShapepi_n/dense_2/Relu*
out_type0*
T0*
_output_shapes
:
f
$pi_n/dense_3/Tensordot/GatherV2/axisConst*
value	B : *
_output_shapes
: *
dtype0
�
pi_n/dense_3/Tensordot/GatherV2GatherV2pi_n/dense_3/Tensordot/Shapepi_n/dense_3/Tensordot/free$pi_n/dense_3/Tensordot/GatherV2/axis*
Taxis0*
Tindices0*

batch_dims *
_output_shapes
:*
Tparams0
h
&pi_n/dense_3/Tensordot/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
!pi_n/dense_3/Tensordot/GatherV2_1GatherV2pi_n/dense_3/Tensordot/Shapepi_n/dense_3/Tensordot/axes&pi_n/dense_3/Tensordot/GatherV2_1/axis*
_output_shapes
:*
Taxis0*

batch_dims *
Tindices0*
Tparams0
f
pi_n/dense_3/Tensordot/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
pi_n/dense_3/Tensordot/ProdProdpi_n/dense_3/Tensordot/GatherV2pi_n/dense_3/Tensordot/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
h
pi_n/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
�
pi_n/dense_3/Tensordot/Prod_1Prod!pi_n/dense_3/Tensordot/GatherV2_1pi_n/dense_3/Tensordot/Const_1*

Tidx0*
_output_shapes
: *
T0*
	keep_dims( 
d
"pi_n/dense_3/Tensordot/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
�
pi_n/dense_3/Tensordot/concatConcatV2pi_n/dense_3/Tensordot/freepi_n/dense_3/Tensordot/axes"pi_n/dense_3/Tensordot/concat/axis*
_output_shapes
:*
T0*

Tidx0*
N
�
pi_n/dense_3/Tensordot/stackPackpi_n/dense_3/Tensordot/Prodpi_n/dense_3/Tensordot/Prod_1*

axis *
T0*
N*
_output_shapes
:
�
 pi_n/dense_3/Tensordot/transpose	Transposepi_n/dense_2/Relupi_n/dense_3/Tensordot/concat*+
_output_shapes
:���������*
T0*
Tperm0
�
pi_n/dense_3/Tensordot/ReshapeReshape pi_n/dense_3/Tensordot/transposepi_n/dense_3/Tensordot/stack*
T0*
Tshape0*0
_output_shapes
:������������������
x
'pi_n/dense_3/Tensordot/transpose_1/permConst*
dtype0*
valueB"       *
_output_shapes
:
�
"pi_n/dense_3/Tensordot/transpose_1	Transposepi_n/dense_3/kernel/read'pi_n/dense_3/Tensordot/transpose_1/perm*
_output_shapes

:*
T0*
Tperm0
w
&pi_n/dense_3/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
�
 pi_n/dense_3/Tensordot/Reshape_1Reshape"pi_n/dense_3/Tensordot/transpose_1&pi_n/dense_3/Tensordot/Reshape_1/shape*
T0*
Tshape0*
_output_shapes

:
�
pi_n/dense_3/Tensordot/MatMulMatMulpi_n/dense_3/Tensordot/Reshape pi_n/dense_3/Tensordot/Reshape_1*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
h
pi_n/dense_3/Tensordot/Const_2Const*
dtype0*
valueB:*
_output_shapes
:
f
$pi_n/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
pi_n/dense_3/Tensordot/concat_1ConcatV2pi_n/dense_3/Tensordot/GatherV2pi_n/dense_3/Tensordot/Const_2$pi_n/dense_3/Tensordot/concat_1/axis*
T0*
N*

Tidx0*
_output_shapes
:
�
pi_n/dense_3/TensordotReshapepi_n/dense_3/Tensordot/MatMulpi_n/dense_3/Tensordot/concat_1*
Tshape0*
T0*+
_output_shapes
:���������
�
pi_n/dense_3/BiasAddBiasAddpi_n/dense_3/Tensordotpi_n/dense_3/bias/read*
data_formatNHWC*+
_output_shapes
:���������*
T0

pi_n/SqueezeSqueezepi_n/dense_3/BiasAdd*'
_output_shapes
:���������*
T0*
squeeze_dims

���������
O

pi_n/sub/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
\
pi_n/subSubPlaceholder_6
pi_n/sub/y*
T0*'
_output_shapes
:���������
O

pi_n/mul/yConst*
_output_shapes
: *
valueB
 * $tI*
dtype0
W
pi_n/mulMulpi_n/sub
pi_n/mul/y*
T0*'
_output_shapes
:���������
Y
pi_n/addAddpi_n/Squeezepi_n/mul*
T0*'
_output_shapes
:���������
Y
pi_n/LogSoftmax
LogSoftmaxpi_n/add*
T0*'
_output_shapes
:���������
j
(pi_n/multinomial/Multinomial/num_samplesConst*
dtype0*
value	B :*
_output_shapes
: 
�
pi_n/multinomial/MultinomialMultinomialpi_n/add(pi_n/multinomial/Multinomial/num_samples*'
_output_shapes
:���������*
T0*
seed�*
seed2�*
output_dtype0	
|
pi_n/Squeeze_1Squeezepi_n/multinomial/Multinomial*
squeeze_dims
*#
_output_shapes
:���������*
T0	
Z
pi_n/one_hot/on_valueConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
[
pi_n/one_hot/off_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
T
pi_n/one_hot/depthConst*
dtype0*
value	B :*
_output_shapes
: 
�
pi_n/one_hotOneHotPlaceholder_4pi_n/one_hot/depthpi_n/one_hot/on_valuepi_n/one_hot/off_value*
TI0*
T0*
axis���������*'
_output_shapes
:���������
b

pi_n/mul_1Mulpi_n/one_hotpi_n/LogSoftmax*'
_output_shapes
:���������*
T0
\
pi_n/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
pi_n/SumSum
pi_n/mul_1pi_n/Sum/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
\
pi_n/one_hot_1/on_valueConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
]
pi_n/one_hot_1/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
V
pi_n/one_hot_1/depthConst*
dtype0*
_output_shapes
: *
value	B :
�
pi_n/one_hot_1OneHotpi_n/Squeeze_1pi_n/one_hot_1/depthpi_n/one_hot_1/on_valuepi_n/one_hot_1/off_value*
axis���������*'
_output_shapes
:���������*
T0*
TI0	
d

pi_n/mul_2Mulpi_n/one_hot_1pi_n/LogSoftmax*'
_output_shapes
:���������*
T0
^
pi_n/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
�

pi_n/Sum_1Sum
pi_n/mul_2pi_n/Sum_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
Q
subSubpi_j/SumPlaceholder_9*#
_output_shapes
:���������*
T0
=
ExpExpsub*
T0*#
_output_shapes
:���������
T
sub_1Subpi_n/SumPlaceholder_10*#
_output_shapes
:���������*
T0
A
Exp_1Expsub_1*
T0*#
_output_shapes
:���������
N
	Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
GreaterGreaterPlaceholder_7	Greater/y*#
_output_shapes
:���������*
T0
J
mul/xConst*
valueB
 *���?*
_output_shapes
: *
dtype0
N
mulMulmul/xPlaceholder_7*#
_output_shapes
:���������*
T0
L
mul_1/xConst*
_output_shapes
: *
valueB
 *��L?*
dtype0
R
mul_1Mulmul_1/xPlaceholder_7*
T0*#
_output_shapes
:���������
S
SelectSelectGreatermulmul_1*
T0*#
_output_shapes
:���������
D
addAddExpExp_1*
T0*#
_output_shapes
:���������
N
mul_2MuladdPlaceholder_7*#
_output_shapes
:���������*
T0
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
MeanMeanMinimumConst*
	keep_dims( *

Tidx0*
_output_shapes
: *
T0
1
NegNegMean*
T0*
_output_shapes
: 
T
sub_2SubPlaceholder_8	v/Squeeze*
T0*#
_output_shapes
:���������
J
pow/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
F
powPowsub_2pow/y*
T0*#
_output_shapes
:���������
Q
Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
Z
Mean_1MeanpowConst_1*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
S
sub_3SubPlaceholder_9pi_j/Sum*#
_output_shapes
:���������*
T0
Q
Const_2Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_2Meansub_3Const_2*
_output_shapes
: *

Tidx0*
T0*
	keep_dims( 
D
Neg_1Negpi_j/Sum*
T0*#
_output_shapes
:���������
Q
Const_3Const*
dtype0*
_output_shapes
:*
valueB: 
\
Mean_3MeanNeg_1Const_3*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
P
Greater_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *���?
T
	Greater_1GreaterExpGreater_1/y*
T0*#
_output_shapes
:���������
K
Less/yConst*
dtype0*
_output_shapes
: *
valueB
 *��L?
G
LessLessExpLess/y*
T0*#
_output_shapes
:���������
L
	LogicalOr	LogicalOr	Greater_1Less*#
_output_shapes
:���������
d
CastCast	LogicalOr*#
_output_shapes
:���������*
Truncate( *

SrcT0
*

DstT0
Q
Const_4Const*
_output_shapes
:*
dtype0*
valueB: 
[
Mean_4MeanCastConst_4*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
T
sub_4SubPlaceholder_10pi_n/Sum*#
_output_shapes
:���������*
T0
Q
Const_5Const*
dtype0*
valueB: *
_output_shapes
:
\
Mean_5Meansub_4Const_5*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
D
Neg_2Negpi_n/Sum*#
_output_shapes
:���������*
T0
Q
Const_6Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_6MeanNeg_2Const_6*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
P
Greater_2/yConst*
_output_shapes
: *
valueB
 *���?*
dtype0
V
	Greater_2GreaterExp_1Greater_2/y*#
_output_shapes
:���������*
T0
M
Less_1/yConst*
valueB
 *��L?*
_output_shapes
: *
dtype0
M
Less_1LessExp_1Less_1/y*
T0*#
_output_shapes
:���������
P
LogicalOr_1	LogicalOr	Greater_2Less_1*#
_output_shapes
:���������
h
Cast_1CastLogicalOr_1*#
_output_shapes
:���������*
Truncate( *

DstT0*

SrcT0

Q
Const_7Const*
valueB: *
_output_shapes
:*
dtype0
]
Mean_7MeanCast_1Const_7*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
X
gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
`
gradients/Mean_grad/ShapeShapeMinimum*
_output_shapes
:*
T0*
out_type0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
b
gradients/Mean_grad/Shape_1ShapeMinimum*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
�
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
�
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:���������
a
gradients/Minimum_grad/ShapeShapemul_2*
out_type0*
_output_shapes
:*
T0
d
gradients/Minimum_grad/Shape_1ShapeSelect*
T0*
_output_shapes
:*
out_type0
y
gradients/Minimum_grad/Shape_2Shapegradients/Mean_grad/truediv*
T0*
out_type0*
_output_shapes
:
g
"gradients/Minimum_grad/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *
dtype0
�
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*#
_output_shapes
:���������*
T0*

index_type0
j
 gradients/Minimum_grad/LessEqual	LessEqualmul_2Select*#
_output_shapes
:���������*
T0
�
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/Mean_grad/truedivgradients/Minimum_grad/zeros*
T0*#
_output_shapes
:���������
�
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
�
gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/Mean_grad/truediv*
T0*#
_output_shapes
:���������
�
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
�
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
Tshape0*
T0*#
_output_shapes
:���������
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
�
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*#
_output_shapes
:���������*1
_class'
%#loc:@gradients/Minimum_grad/Reshape*
T0
�
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*#
_output_shapes
:���������*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1*
T0
]
gradients/mul_2_grad/ShapeShapeadd*
out_type0*
_output_shapes
:*
T0
i
gradients/mul_2_grad/Shape_1ShapePlaceholder_7*
_output_shapes
:*
out_type0*
T0
�
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependencyPlaceholder_7*#
_output_shapes
:���������*
T0
�
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
�
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
�
gradients/mul_2_grad/Mul_1Muladd/gradients/Minimum_grad/tuple/control_dependency*#
_output_shapes
:���������*
T0
�
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
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
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*#
_output_shapes
:���������*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*
T0
�
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*#
_output_shapes
:���������*
T0
[
gradients/add_grad/ShapeShapeExp*
T0*
_output_shapes
:*
out_type0
_
gradients/add_grad/Shape_1ShapeExp_1*
_output_shapes
:*
T0*
out_type0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSum-gradients/mul_2_grad/tuple/control_dependency(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
�
gradients/add_grad/Sum_1Sum-gradients/mul_2_grad/tuple/control_dependency*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*#
_output_shapes
:���������*
Tshape0*
T0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*-
_class#
!loc:@gradients/add_grad/Reshape*#
_output_shapes
:���������*
T0
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
T0*#
_output_shapes
:���������
}
gradients/Exp_grad/mulMul+gradients/add_grad/tuple/control_dependencyExp*#
_output_shapes
:���������*
T0
�
gradients/Exp_1_grad/mulMul-gradients/add_grad/tuple/control_dependency_1Exp_1*
T0*#
_output_shapes
:���������
`
gradients/sub_grad/ShapeShapepi_j/Sum*
T0*
_output_shapes
:*
out_type0
g
gradients/sub_grad/Shape_1ShapePlaceholder_9*
out_type0*
_output_shapes
:*
T0
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:*

Tidx0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
�
gradients/sub_grad/Sum_1Sumgradients/Exp_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*#
_output_shapes
:���������*
Tshape0*
T0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*#
_output_shapes
:���������
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*#
_output_shapes
:���������*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
b
gradients/sub_1_grad/ShapeShapepi_n/Sum*
_output_shapes
:*
T0*
out_type0
j
gradients/sub_1_grad/Shape_1ShapePlaceholder_10*
_output_shapes
:*
out_type0*
T0
�
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_1_grad/SumSumgradients/Exp_1_grad/mul*gradients/sub_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
�
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
Tshape0*
T0*#
_output_shapes
:���������
�
gradients/sub_1_grad/Sum_1Sumgradients/Exp_1_grad/mul,gradients/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
^
gradients/sub_1_grad/NegNeggradients/sub_1_grad/Sum_1*
T0*
_output_shapes
:
�
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Neggradients/sub_1_grad/Shape_1*
T0*#
_output_shapes
:���������*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
�
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*#
_output_shapes
:���������*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*#
_output_shapes
:���������*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1*
T0
g
gradients/pi_j/Sum_grad/ShapeShape
pi_j/mul_1*
_output_shapes
:*
T0*
out_type0
�
gradients/pi_j/Sum_grad/SizeConst*
value	B :*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients/pi_j/Sum_grad/addAddpi_j/Sum/reduction_indicesgradients/pi_j/Sum_grad/Size*
_output_shapes
: *0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape*
T0
�
gradients/pi_j/Sum_grad/modFloorModgradients/pi_j/Sum_grad/addgradients/pi_j/Sum_grad/Size*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape
�
gradients/pi_j/Sum_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape
�
#gradients/pi_j/Sum_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: *0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape
�
#gradients/pi_j/Sum_grad/range/deltaConst*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
�
gradients/pi_j/Sum_grad/rangeRange#gradients/pi_j/Sum_grad/range/startgradients/pi_j/Sum_grad/Size#gradients/pi_j/Sum_grad/range/delta*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape*

Tidx0*
_output_shapes
:
�
"gradients/pi_j/Sum_grad/Fill/valueConst*
dtype0*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape
�
gradients/pi_j/Sum_grad/FillFillgradients/pi_j/Sum_grad/Shape_1"gradients/pi_j/Sum_grad/Fill/value*
T0*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape*
_output_shapes
: *

index_type0
�
%gradients/pi_j/Sum_grad/DynamicStitchDynamicStitchgradients/pi_j/Sum_grad/rangegradients/pi_j/Sum_grad/modgradients/pi_j/Sum_grad/Shapegradients/pi_j/Sum_grad/Fill*
N*
_output_shapes
:*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape*
T0
�
!gradients/pi_j/Sum_grad/Maximum/yConst*
dtype0*
value	B :*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape*
_output_shapes
: 
�
gradients/pi_j/Sum_grad/MaximumMaximum%gradients/pi_j/Sum_grad/DynamicStitch!gradients/pi_j/Sum_grad/Maximum/y*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape*
_output_shapes
:*
T0
�
 gradients/pi_j/Sum_grad/floordivFloorDivgradients/pi_j/Sum_grad/Shapegradients/pi_j/Sum_grad/Maximum*
T0*0
_class&
$"loc:@gradients/pi_j/Sum_grad/Shape*
_output_shapes
:
�
gradients/pi_j/Sum_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency%gradients/pi_j/Sum_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0
�
gradients/pi_j/Sum_grad/TileTilegradients/pi_j/Sum_grad/Reshape gradients/pi_j/Sum_grad/floordiv*(
_output_shapes
:����������*
T0*

Tmultiples0
g
gradients/pi_n/Sum_grad/ShapeShape
pi_n/mul_1*
out_type0*
T0*
_output_shapes
:
�
gradients/pi_n/Sum_grad/SizeConst*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
gradients/pi_n/Sum_grad/addAddpi_n/Sum/reduction_indicesgradients/pi_n/Sum_grad/Size*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
T0*
_output_shapes
: 
�
gradients/pi_n/Sum_grad/modFloorModgradients/pi_n/Sum_grad/addgradients/pi_n/Sum_grad/Size*
_output_shapes
: *0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
T0
�
gradients/pi_n/Sum_grad/Shape_1Const*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
�
#gradients/pi_n/Sum_grad/range/startConst*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
dtype0*
_output_shapes
: *
value	B : 
�
#gradients/pi_n/Sum_grad/range/deltaConst*
_output_shapes
: *
value	B :*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
dtype0
�
gradients/pi_n/Sum_grad/rangeRange#gradients/pi_n/Sum_grad/range/startgradients/pi_n/Sum_grad/Size#gradients/pi_n/Sum_grad/range/delta*
_output_shapes
:*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*

Tidx0
�
"gradients/pi_n/Sum_grad/Fill/valueConst*
_output_shapes
: *0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
dtype0*
value	B :
�
gradients/pi_n/Sum_grad/FillFillgradients/pi_n/Sum_grad/Shape_1"gradients/pi_n/Sum_grad/Fill/value*
_output_shapes
: *0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*

index_type0*
T0
�
%gradients/pi_n/Sum_grad/DynamicStitchDynamicStitchgradients/pi_n/Sum_grad/rangegradients/pi_n/Sum_grad/modgradients/pi_n/Sum_grad/Shapegradients/pi_n/Sum_grad/Fill*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
_output_shapes
:*
T0*
N
�
!gradients/pi_n/Sum_grad/Maximum/yConst*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
�
gradients/pi_n/Sum_grad/MaximumMaximum%gradients/pi_n/Sum_grad/DynamicStitch!gradients/pi_n/Sum_grad/Maximum/y*
_output_shapes
:*
T0*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape
�
 gradients/pi_n/Sum_grad/floordivFloorDivgradients/pi_n/Sum_grad/Shapegradients/pi_n/Sum_grad/Maximum*0
_class&
$"loc:@gradients/pi_n/Sum_grad/Shape*
_output_shapes
:*
T0
�
gradients/pi_n/Sum_grad/ReshapeReshape-gradients/sub_1_grad/tuple/control_dependency%gradients/pi_n/Sum_grad/DynamicStitch*
Tshape0*
T0*0
_output_shapes
:������������������
�
gradients/pi_n/Sum_grad/TileTilegradients/pi_n/Sum_grad/Reshape gradients/pi_n/Sum_grad/floordiv*'
_output_shapes
:���������*
T0*

Tmultiples0
k
gradients/pi_j/mul_1_grad/ShapeShapepi_j/one_hot*
_output_shapes
:*
out_type0*
T0
p
!gradients/pi_j/mul_1_grad/Shape_1Shapepi_j/LogSoftmax*
out_type0*
_output_shapes
:*
T0
�
/gradients/pi_j/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi_j/mul_1_grad/Shape!gradients/pi_j/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/pi_j/mul_1_grad/MulMulgradients/pi_j/Sum_grad/Tilepi_j/LogSoftmax*
T0*(
_output_shapes
:����������
�
gradients/pi_j/mul_1_grad/SumSumgradients/pi_j/mul_1_grad/Mul/gradients/pi_j/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
�
!gradients/pi_j/mul_1_grad/ReshapeReshapegradients/pi_j/mul_1_grad/Sumgradients/pi_j/mul_1_grad/Shape*
T0*
Tshape0*(
_output_shapes
:����������
�
gradients/pi_j/mul_1_grad/Mul_1Mulpi_j/one_hotgradients/pi_j/Sum_grad/Tile*
T0*(
_output_shapes
:����������
�
gradients/pi_j/mul_1_grad/Sum_1Sumgradients/pi_j/mul_1_grad/Mul_11gradients/pi_j/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
�
#gradients/pi_j/mul_1_grad/Reshape_1Reshapegradients/pi_j/mul_1_grad/Sum_1!gradients/pi_j/mul_1_grad/Shape_1*
Tshape0*
T0*(
_output_shapes
:����������
|
*gradients/pi_j/mul_1_grad/tuple/group_depsNoOp"^gradients/pi_j/mul_1_grad/Reshape$^gradients/pi_j/mul_1_grad/Reshape_1
�
2gradients/pi_j/mul_1_grad/tuple/control_dependencyIdentity!gradients/pi_j/mul_1_grad/Reshape+^gradients/pi_j/mul_1_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients/pi_j/mul_1_grad/Reshape*(
_output_shapes
:����������
�
4gradients/pi_j/mul_1_grad/tuple/control_dependency_1Identity#gradients/pi_j/mul_1_grad/Reshape_1+^gradients/pi_j/mul_1_grad/tuple/group_deps*
T0*(
_output_shapes
:����������*6
_class,
*(loc:@gradients/pi_j/mul_1_grad/Reshape_1
k
gradients/pi_n/mul_1_grad/ShapeShapepi_n/one_hot*
_output_shapes
:*
T0*
out_type0
p
!gradients/pi_n/mul_1_grad/Shape_1Shapepi_n/LogSoftmax*
_output_shapes
:*
out_type0*
T0
�
/gradients/pi_n/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi_n/mul_1_grad/Shape!gradients/pi_n/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients/pi_n/mul_1_grad/MulMulgradients/pi_n/Sum_grad/Tilepi_n/LogSoftmax*'
_output_shapes
:���������*
T0
�
gradients/pi_n/mul_1_grad/SumSumgradients/pi_n/mul_1_grad/Mul/gradients/pi_n/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
!gradients/pi_n/mul_1_grad/ReshapeReshapegradients/pi_n/mul_1_grad/Sumgradients/pi_n/mul_1_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
gradients/pi_n/mul_1_grad/Mul_1Mulpi_n/one_hotgradients/pi_n/Sum_grad/Tile*'
_output_shapes
:���������*
T0
�
gradients/pi_n/mul_1_grad/Sum_1Sumgradients/pi_n/mul_1_grad/Mul_11gradients/pi_n/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
#gradients/pi_n/mul_1_grad/Reshape_1Reshapegradients/pi_n/mul_1_grad/Sum_1!gradients/pi_n/mul_1_grad/Shape_1*
Tshape0*
T0*'
_output_shapes
:���������
|
*gradients/pi_n/mul_1_grad/tuple/group_depsNoOp"^gradients/pi_n/mul_1_grad/Reshape$^gradients/pi_n/mul_1_grad/Reshape_1
�
2gradients/pi_n/mul_1_grad/tuple/control_dependencyIdentity!gradients/pi_n/mul_1_grad/Reshape+^gradients/pi_n/mul_1_grad/tuple/group_deps*4
_class*
(&loc:@gradients/pi_n/mul_1_grad/Reshape*'
_output_shapes
:���������*
T0
�
4gradients/pi_n/mul_1_grad/tuple/control_dependency_1Identity#gradients/pi_n/mul_1_grad/Reshape_1+^gradients/pi_n/mul_1_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*6
_class,
*(loc:@gradients/pi_n/mul_1_grad/Reshape_1
m
"gradients/pi_j/LogSoftmax_grad/ExpExppi_j/LogSoftmax*(
_output_shapes
:����������*
T0

4gradients/pi_j/LogSoftmax_grad/Sum/reduction_indicesConst*
valueB :
���������*
_output_shapes
: *
dtype0
�
"gradients/pi_j/LogSoftmax_grad/SumSum4gradients/pi_j/mul_1_grad/tuple/control_dependency_14gradients/pi_j/LogSoftmax_grad/Sum/reduction_indices*'
_output_shapes
:���������*
T0*
	keep_dims(*

Tidx0
�
"gradients/pi_j/LogSoftmax_grad/mulMul"gradients/pi_j/LogSoftmax_grad/Sum"gradients/pi_j/LogSoftmax_grad/Exp*
T0*(
_output_shapes
:����������
�
"gradients/pi_j/LogSoftmax_grad/subSub4gradients/pi_j/mul_1_grad/tuple/control_dependency_1"gradients/pi_j/LogSoftmax_grad/mul*(
_output_shapes
:����������*
T0
l
"gradients/pi_n/LogSoftmax_grad/ExpExppi_n/LogSoftmax*
T0*'
_output_shapes
:���������

4gradients/pi_n/LogSoftmax_grad/Sum/reduction_indicesConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
"gradients/pi_n/LogSoftmax_grad/SumSum4gradients/pi_n/mul_1_grad/tuple/control_dependency_14gradients/pi_n/LogSoftmax_grad/Sum/reduction_indices*
	keep_dims(*
T0*'
_output_shapes
:���������*

Tidx0
�
"gradients/pi_n/LogSoftmax_grad/mulMul"gradients/pi_n/LogSoftmax_grad/Sum"gradients/pi_n/LogSoftmax_grad/Exp*'
_output_shapes
:���������*
T0
�
"gradients/pi_n/LogSoftmax_grad/subSub4gradients/pi_n/mul_1_grad/tuple/control_dependency_1"gradients/pi_n/LogSoftmax_grad/mul*'
_output_shapes
:���������*
T0
i
gradients/pi_j/add_grad/ShapeShapepi_j/Squeeze*
T0*
_output_shapes
:*
out_type0
g
gradients/pi_j/add_grad/Shape_1Shapepi_j/mul*
T0*
_output_shapes
:*
out_type0
�
-gradients/pi_j/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi_j/add_grad/Shapegradients/pi_j/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/pi_j/add_grad/SumSum"gradients/pi_j/LogSoftmax_grad/sub-gradients/pi_j/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/pi_j/add_grad/ReshapeReshapegradients/pi_j/add_grad/Sumgradients/pi_j/add_grad/Shape*
T0*(
_output_shapes
:����������*
Tshape0
�
gradients/pi_j/add_grad/Sum_1Sum"gradients/pi_j/LogSoftmax_grad/sub/gradients/pi_j/add_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
�
!gradients/pi_j/add_grad/Reshape_1Reshapegradients/pi_j/add_grad/Sum_1gradients/pi_j/add_grad/Shape_1*
T0*
Tshape0*(
_output_shapes
:����������
v
(gradients/pi_j/add_grad/tuple/group_depsNoOp ^gradients/pi_j/add_grad/Reshape"^gradients/pi_j/add_grad/Reshape_1
�
0gradients/pi_j/add_grad/tuple/control_dependencyIdentitygradients/pi_j/add_grad/Reshape)^gradients/pi_j/add_grad/tuple/group_deps*(
_output_shapes
:����������*2
_class(
&$loc:@gradients/pi_j/add_grad/Reshape*
T0
�
2gradients/pi_j/add_grad/tuple/control_dependency_1Identity!gradients/pi_j/add_grad/Reshape_1)^gradients/pi_j/add_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*4
_class*
(&loc:@gradients/pi_j/add_grad/Reshape_1
i
gradients/pi_n/add_grad/ShapeShapepi_n/Squeeze*
out_type0*
T0*
_output_shapes
:
g
gradients/pi_n/add_grad/Shape_1Shapepi_n/mul*
out_type0*
T0*
_output_shapes
:
�
-gradients/pi_n/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi_n/add_grad/Shapegradients/pi_n/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/pi_n/add_grad/SumSum"gradients/pi_n/LogSoftmax_grad/sub-gradients/pi_n/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
�
gradients/pi_n/add_grad/ReshapeReshapegradients/pi_n/add_grad/Sumgradients/pi_n/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
gradients/pi_n/add_grad/Sum_1Sum"gradients/pi_n/LogSoftmax_grad/sub/gradients/pi_n/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
�
!gradients/pi_n/add_grad/Reshape_1Reshapegradients/pi_n/add_grad/Sum_1gradients/pi_n/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
v
(gradients/pi_n/add_grad/tuple/group_depsNoOp ^gradients/pi_n/add_grad/Reshape"^gradients/pi_n/add_grad/Reshape_1
�
0gradients/pi_n/add_grad/tuple/control_dependencyIdentitygradients/pi_n/add_grad/Reshape)^gradients/pi_n/add_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*2
_class(
&$loc:@gradients/pi_n/add_grad/Reshape
�
2gradients/pi_n/add_grad/tuple/control_dependency_1Identity!gradients/pi_n/add_grad/Reshape_1)^gradients/pi_n/add_grad/tuple/group_deps*'
_output_shapes
:���������*4
_class*
(&loc:@gradients/pi_n/add_grad/Reshape_1*
T0
u
!gradients/pi_j/Squeeze_grad/ShapeShapepi_j/dense_3/BiasAdd*
_output_shapes
:*
T0*
out_type0
�
#gradients/pi_j/Squeeze_grad/ReshapeReshape0gradients/pi_j/add_grad/tuple/control_dependency!gradients/pi_j/Squeeze_grad/Shape*
Tshape0*
T0*,
_output_shapes
:����������
u
!gradients/pi_n/Squeeze_grad/ShapeShapepi_n/dense_3/BiasAdd*
_output_shapes
:*
out_type0*
T0
�
#gradients/pi_n/Squeeze_grad/ReshapeReshape0gradients/pi_n/add_grad/tuple/control_dependency!gradients/pi_n/Squeeze_grad/Shape*
Tshape0*
T0*+
_output_shapes
:���������
�
/gradients/pi_j/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/pi_j/Squeeze_grad/Reshape*
T0*
data_formatNHWC*
_output_shapes
:
�
4gradients/pi_j/dense_3/BiasAdd_grad/tuple/group_depsNoOp$^gradients/pi_j/Squeeze_grad/Reshape0^gradients/pi_j/dense_3/BiasAdd_grad/BiasAddGrad
�
<gradients/pi_j/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/pi_j/Squeeze_grad/Reshape5^gradients/pi_j/dense_3/BiasAdd_grad/tuple/group_deps*
T0*,
_output_shapes
:����������*6
_class,
*(loc:@gradients/pi_j/Squeeze_grad/Reshape
�
>gradients/pi_j/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/pi_j/dense_3/BiasAdd_grad/BiasAddGrad5^gradients/pi_j/dense_3/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*B
_class8
64loc:@gradients/pi_j/dense_3/BiasAdd_grad/BiasAddGrad*
T0
�
/gradients/pi_n/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/pi_n/Squeeze_grad/Reshape*
_output_shapes
:*
data_formatNHWC*
T0
�
4gradients/pi_n/dense_3/BiasAdd_grad/tuple/group_depsNoOp$^gradients/pi_n/Squeeze_grad/Reshape0^gradients/pi_n/dense_3/BiasAdd_grad/BiasAddGrad
�
<gradients/pi_n/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/pi_n/Squeeze_grad/Reshape5^gradients/pi_n/dense_3/BiasAdd_grad/tuple/group_deps*+
_output_shapes
:���������*
T0*6
_class,
*(loc:@gradients/pi_n/Squeeze_grad/Reshape
�
>gradients/pi_n/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/pi_n/dense_3/BiasAdd_grad/BiasAddGrad5^gradients/pi_n/dense_3/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients/pi_n/dense_3/BiasAdd_grad/BiasAddGrad
�
+gradients/pi_j/dense_3/Tensordot_grad/ShapeShapepi_j/dense_3/Tensordot/MatMul*
T0*
_output_shapes
:*
out_type0
�
-gradients/pi_j/dense_3/Tensordot_grad/ReshapeReshape<gradients/pi_j/dense_3/BiasAdd_grad/tuple/control_dependency+gradients/pi_j/dense_3/Tensordot_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
+gradients/pi_n/dense_3/Tensordot_grad/ShapeShapepi_n/dense_3/Tensordot/MatMul*
out_type0*
_output_shapes
:*
T0
�
-gradients/pi_n/dense_3/Tensordot_grad/ReshapeReshape<gradients/pi_n/dense_3/BiasAdd_grad/tuple/control_dependency+gradients/pi_n/dense_3/Tensordot_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
3gradients/pi_j/dense_3/Tensordot/MatMul_grad/MatMulMatMul-gradients/pi_j/dense_3/Tensordot_grad/Reshape pi_j/dense_3/Tensordot/Reshape_1*
transpose_a( *
T0*
transpose_b(*'
_output_shapes
:���������
�
5gradients/pi_j/dense_3/Tensordot/MatMul_grad/MatMul_1MatMulpi_j/dense_3/Tensordot/Reshape-gradients/pi_j/dense_3/Tensordot_grad/Reshape*
transpose_a(*
transpose_b( *'
_output_shapes
:���������*
T0
�
=gradients/pi_j/dense_3/Tensordot/MatMul_grad/tuple/group_depsNoOp4^gradients/pi_j/dense_3/Tensordot/MatMul_grad/MatMul6^gradients/pi_j/dense_3/Tensordot/MatMul_grad/MatMul_1
�
Egradients/pi_j/dense_3/Tensordot/MatMul_grad/tuple/control_dependencyIdentity3gradients/pi_j/dense_3/Tensordot/MatMul_grad/MatMul>^gradients/pi_j/dense_3/Tensordot/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/pi_j/dense_3/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:���������*
T0
�
Ggradients/pi_j/dense_3/Tensordot/MatMul_grad/tuple/control_dependency_1Identity5gradients/pi_j/dense_3/Tensordot/MatMul_grad/MatMul_1>^gradients/pi_j/dense_3/Tensordot/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/pi_j/dense_3/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:
�
3gradients/pi_n/dense_3/Tensordot/MatMul_grad/MatMulMatMul-gradients/pi_n/dense_3/Tensordot_grad/Reshape pi_n/dense_3/Tensordot/Reshape_1*
transpose_a( *
T0*'
_output_shapes
:���������*
transpose_b(
�
5gradients/pi_n/dense_3/Tensordot/MatMul_grad/MatMul_1MatMulpi_n/dense_3/Tensordot/Reshape-gradients/pi_n/dense_3/Tensordot_grad/Reshape*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a(
�
=gradients/pi_n/dense_3/Tensordot/MatMul_grad/tuple/group_depsNoOp4^gradients/pi_n/dense_3/Tensordot/MatMul_grad/MatMul6^gradients/pi_n/dense_3/Tensordot/MatMul_grad/MatMul_1
�
Egradients/pi_n/dense_3/Tensordot/MatMul_grad/tuple/control_dependencyIdentity3gradients/pi_n/dense_3/Tensordot/MatMul_grad/MatMul>^gradients/pi_n/dense_3/Tensordot/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*F
_class<
:8loc:@gradients/pi_n/dense_3/Tensordot/MatMul_grad/MatMul
�
Ggradients/pi_n/dense_3/Tensordot/MatMul_grad/tuple/control_dependency_1Identity5gradients/pi_n/dense_3/Tensordot/MatMul_grad/MatMul_1>^gradients/pi_n/dense_3/Tensordot/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/pi_n/dense_3/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:
�
3gradients/pi_j/dense_3/Tensordot/Reshape_grad/ShapeShape pi_j/dense_3/Tensordot/transpose*
out_type0*
_output_shapes
:*
T0
�
5gradients/pi_j/dense_3/Tensordot/Reshape_grad/ReshapeReshapeEgradients/pi_j/dense_3/Tensordot/MatMul_grad/tuple/control_dependency3gradients/pi_j/dense_3/Tensordot/Reshape_grad/Shape*
Tshape0*
T0*,
_output_shapes
:����������
�
5gradients/pi_j/dense_3/Tensordot/Reshape_1_grad/ShapeConst*
valueB"      *
_output_shapes
:*
dtype0
�
7gradients/pi_j/dense_3/Tensordot/Reshape_1_grad/ReshapeReshapeGgradients/pi_j/dense_3/Tensordot/MatMul_grad/tuple/control_dependency_15gradients/pi_j/dense_3/Tensordot/Reshape_1_grad/Shape*
T0*
_output_shapes

:*
Tshape0
�
3gradients/pi_n/dense_3/Tensordot/Reshape_grad/ShapeShape pi_n/dense_3/Tensordot/transpose*
out_type0*
_output_shapes
:*
T0
�
5gradients/pi_n/dense_3/Tensordot/Reshape_grad/ReshapeReshapeEgradients/pi_n/dense_3/Tensordot/MatMul_grad/tuple/control_dependency3gradients/pi_n/dense_3/Tensordot/Reshape_grad/Shape*
Tshape0*
T0*+
_output_shapes
:���������
�
5gradients/pi_n/dense_3/Tensordot/Reshape_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
7gradients/pi_n/dense_3/Tensordot/Reshape_1_grad/ReshapeReshapeGgradients/pi_n/dense_3/Tensordot/MatMul_grad/tuple/control_dependency_15gradients/pi_n/dense_3/Tensordot/Reshape_1_grad/Shape*
_output_shapes

:*
Tshape0*
T0
�
Agradients/pi_j/dense_3/Tensordot/transpose_grad/InvertPermutationInvertPermutationpi_j/dense_3/Tensordot/concat*
T0*
_output_shapes
:
�
9gradients/pi_j/dense_3/Tensordot/transpose_grad/transpose	Transpose5gradients/pi_j/dense_3/Tensordot/Reshape_grad/ReshapeAgradients/pi_j/dense_3/Tensordot/transpose_grad/InvertPermutation*,
_output_shapes
:����������*
T0*
Tperm0
�
Cgradients/pi_j/dense_3/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation'pi_j/dense_3/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
;gradients/pi_j/dense_3/Tensordot/transpose_1_grad/transpose	Transpose7gradients/pi_j/dense_3/Tensordot/Reshape_1_grad/ReshapeCgradients/pi_j/dense_3/Tensordot/transpose_1_grad/InvertPermutation*
_output_shapes

:*
Tperm0*
T0
�
Agradients/pi_n/dense_3/Tensordot/transpose_grad/InvertPermutationInvertPermutationpi_n/dense_3/Tensordot/concat*
_output_shapes
:*
T0
�
9gradients/pi_n/dense_3/Tensordot/transpose_grad/transpose	Transpose5gradients/pi_n/dense_3/Tensordot/Reshape_grad/ReshapeAgradients/pi_n/dense_3/Tensordot/transpose_grad/InvertPermutation*
T0*+
_output_shapes
:���������*
Tperm0
�
Cgradients/pi_n/dense_3/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation'pi_n/dense_3/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
;gradients/pi_n/dense_3/Tensordot/transpose_1_grad/transpose	Transpose7gradients/pi_n/dense_3/Tensordot/Reshape_1_grad/ReshapeCgradients/pi_n/dense_3/Tensordot/transpose_1_grad/InvertPermutation*
_output_shapes

:*
Tperm0*
T0
�
)gradients/pi_j/dense_2/Relu_grad/ReluGradReluGrad9gradients/pi_j/dense_3/Tensordot/transpose_grad/transposepi_j/dense_2/Relu*,
_output_shapes
:����������*
T0
�
)gradients/pi_n/dense_2/Relu_grad/ReluGradReluGrad9gradients/pi_n/dense_3/Tensordot/transpose_grad/transposepi_n/dense_2/Relu*
T0*+
_output_shapes
:���������
�
/gradients/pi_j/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/pi_j/dense_2/Relu_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
�
4gradients/pi_j/dense_2/BiasAdd_grad/tuple/group_depsNoOp0^gradients/pi_j/dense_2/BiasAdd_grad/BiasAddGrad*^gradients/pi_j/dense_2/Relu_grad/ReluGrad
�
<gradients/pi_j/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/pi_j/dense_2/Relu_grad/ReluGrad5^gradients/pi_j/dense_2/BiasAdd_grad/tuple/group_deps*
T0*,
_output_shapes
:����������*<
_class2
0.loc:@gradients/pi_j/dense_2/Relu_grad/ReluGrad
�
>gradients/pi_j/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/pi_j/dense_2/BiasAdd_grad/BiasAddGrad5^gradients/pi_j/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients/pi_j/dense_2/BiasAdd_grad/BiasAddGrad
�
/gradients/pi_n/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/pi_n/dense_2/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
�
4gradients/pi_n/dense_2/BiasAdd_grad/tuple/group_depsNoOp0^gradients/pi_n/dense_2/BiasAdd_grad/BiasAddGrad*^gradients/pi_n/dense_2/Relu_grad/ReluGrad
�
<gradients/pi_n/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/pi_n/dense_2/Relu_grad/ReluGrad5^gradients/pi_n/dense_2/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients/pi_n/dense_2/Relu_grad/ReluGrad*+
_output_shapes
:���������*
T0
�
>gradients/pi_n/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/pi_n/dense_2/BiasAdd_grad/BiasAddGrad5^gradients/pi_n/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*B
_class8
64loc:@gradients/pi_n/dense_2/BiasAdd_grad/BiasAddGrad*
T0
�
+gradients/pi_j/dense_2/Tensordot_grad/ShapeShapepi_j/dense_2/Tensordot/MatMul*
_output_shapes
:*
out_type0*
T0
�
-gradients/pi_j/dense_2/Tensordot_grad/ReshapeReshape<gradients/pi_j/dense_2/BiasAdd_grad/tuple/control_dependency+gradients/pi_j/dense_2/Tensordot_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
+gradients/pi_n/dense_2/Tensordot_grad/ShapeShapepi_n/dense_2/Tensordot/MatMul*
T0*
_output_shapes
:*
out_type0
�
-gradients/pi_n/dense_2/Tensordot_grad/ReshapeReshape<gradients/pi_n/dense_2/BiasAdd_grad/tuple/control_dependency+gradients/pi_n/dense_2/Tensordot_grad/Shape*'
_output_shapes
:���������*
Tshape0*
T0
�
3gradients/pi_j/dense_2/Tensordot/MatMul_grad/MatMulMatMul-gradients/pi_j/dense_2/Tensordot_grad/Reshape pi_j/dense_2/Tensordot/Reshape_1*
T0*
transpose_a( *
transpose_b(*'
_output_shapes
:���������
�
5gradients/pi_j/dense_2/Tensordot/MatMul_grad/MatMul_1MatMulpi_j/dense_2/Tensordot/Reshape-gradients/pi_j/dense_2/Tensordot_grad/Reshape*
transpose_b( *'
_output_shapes
:���������*
transpose_a(*
T0
�
=gradients/pi_j/dense_2/Tensordot/MatMul_grad/tuple/group_depsNoOp4^gradients/pi_j/dense_2/Tensordot/MatMul_grad/MatMul6^gradients/pi_j/dense_2/Tensordot/MatMul_grad/MatMul_1
�
Egradients/pi_j/dense_2/Tensordot/MatMul_grad/tuple/control_dependencyIdentity3gradients/pi_j/dense_2/Tensordot/MatMul_grad/MatMul>^gradients/pi_j/dense_2/Tensordot/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*F
_class<
:8loc:@gradients/pi_j/dense_2/Tensordot/MatMul_grad/MatMul
�
Ggradients/pi_j/dense_2/Tensordot/MatMul_grad/tuple/control_dependency_1Identity5gradients/pi_j/dense_2/Tensordot/MatMul_grad/MatMul_1>^gradients/pi_j/dense_2/Tensordot/MatMul_grad/tuple/group_deps*
T0*
_output_shapes

:*H
_class>
<:loc:@gradients/pi_j/dense_2/Tensordot/MatMul_grad/MatMul_1
�
3gradients/pi_n/dense_2/Tensordot/MatMul_grad/MatMulMatMul-gradients/pi_n/dense_2/Tensordot_grad/Reshape pi_n/dense_2/Tensordot/Reshape_1*
T0*'
_output_shapes
:���������*
transpose_b(*
transpose_a( 
�
5gradients/pi_n/dense_2/Tensordot/MatMul_grad/MatMul_1MatMulpi_n/dense_2/Tensordot/Reshape-gradients/pi_n/dense_2/Tensordot_grad/Reshape*
transpose_a(*
transpose_b( *'
_output_shapes
:���������*
T0
�
=gradients/pi_n/dense_2/Tensordot/MatMul_grad/tuple/group_depsNoOp4^gradients/pi_n/dense_2/Tensordot/MatMul_grad/MatMul6^gradients/pi_n/dense_2/Tensordot/MatMul_grad/MatMul_1
�
Egradients/pi_n/dense_2/Tensordot/MatMul_grad/tuple/control_dependencyIdentity3gradients/pi_n/dense_2/Tensordot/MatMul_grad/MatMul>^gradients/pi_n/dense_2/Tensordot/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*F
_class<
:8loc:@gradients/pi_n/dense_2/Tensordot/MatMul_grad/MatMul*
T0
�
Ggradients/pi_n/dense_2/Tensordot/MatMul_grad/tuple/control_dependency_1Identity5gradients/pi_n/dense_2/Tensordot/MatMul_grad/MatMul_1>^gradients/pi_n/dense_2/Tensordot/MatMul_grad/tuple/group_deps*H
_class>
<:loc:@gradients/pi_n/dense_2/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
�
3gradients/pi_j/dense_2/Tensordot/Reshape_grad/ShapeShape pi_j/dense_2/Tensordot/transpose*
_output_shapes
:*
T0*
out_type0
�
5gradients/pi_j/dense_2/Tensordot/Reshape_grad/ReshapeReshapeEgradients/pi_j/dense_2/Tensordot/MatMul_grad/tuple/control_dependency3gradients/pi_j/dense_2/Tensordot/Reshape_grad/Shape*
Tshape0*
T0*,
_output_shapes
:����������
�
5gradients/pi_j/dense_2/Tensordot/Reshape_1_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
7gradients/pi_j/dense_2/Tensordot/Reshape_1_grad/ReshapeReshapeGgradients/pi_j/dense_2/Tensordot/MatMul_grad/tuple/control_dependency_15gradients/pi_j/dense_2/Tensordot/Reshape_1_grad/Shape*
_output_shapes

:*
Tshape0*
T0
�
3gradients/pi_n/dense_2/Tensordot/Reshape_grad/ShapeShape pi_n/dense_2/Tensordot/transpose*
_output_shapes
:*
T0*
out_type0
�
5gradients/pi_n/dense_2/Tensordot/Reshape_grad/ReshapeReshapeEgradients/pi_n/dense_2/Tensordot/MatMul_grad/tuple/control_dependency3gradients/pi_n/dense_2/Tensordot/Reshape_grad/Shape*
T0*+
_output_shapes
:���������*
Tshape0
�
5gradients/pi_n/dense_2/Tensordot/Reshape_1_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
�
7gradients/pi_n/dense_2/Tensordot/Reshape_1_grad/ReshapeReshapeGgradients/pi_n/dense_2/Tensordot/MatMul_grad/tuple/control_dependency_15gradients/pi_n/dense_2/Tensordot/Reshape_1_grad/Shape*
Tshape0*
_output_shapes

:*
T0
�
Agradients/pi_j/dense_2/Tensordot/transpose_grad/InvertPermutationInvertPermutationpi_j/dense_2/Tensordot/concat*
_output_shapes
:*
T0
�
9gradients/pi_j/dense_2/Tensordot/transpose_grad/transpose	Transpose5gradients/pi_j/dense_2/Tensordot/Reshape_grad/ReshapeAgradients/pi_j/dense_2/Tensordot/transpose_grad/InvertPermutation*
Tperm0*,
_output_shapes
:����������*
T0
�
Cgradients/pi_j/dense_2/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation'pi_j/dense_2/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
;gradients/pi_j/dense_2/Tensordot/transpose_1_grad/transpose	Transpose7gradients/pi_j/dense_2/Tensordot/Reshape_1_grad/ReshapeCgradients/pi_j/dense_2/Tensordot/transpose_1_grad/InvertPermutation*
T0*
Tperm0*
_output_shapes

:
�
Agradients/pi_n/dense_2/Tensordot/transpose_grad/InvertPermutationInvertPermutationpi_n/dense_2/Tensordot/concat*
_output_shapes
:*
T0
�
9gradients/pi_n/dense_2/Tensordot/transpose_grad/transpose	Transpose5gradients/pi_n/dense_2/Tensordot/Reshape_grad/ReshapeAgradients/pi_n/dense_2/Tensordot/transpose_grad/InvertPermutation*
Tperm0*+
_output_shapes
:���������*
T0
�
Cgradients/pi_n/dense_2/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation'pi_n/dense_2/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
�
;gradients/pi_n/dense_2/Tensordot/transpose_1_grad/transpose	Transpose7gradients/pi_n/dense_2/Tensordot/Reshape_1_grad/ReshapeCgradients/pi_n/dense_2/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
_output_shapes

:*
T0
�
)gradients/pi_j/dense_1/Relu_grad/ReluGradReluGrad9gradients/pi_j/dense_2/Tensordot/transpose_grad/transposepi_j/dense_1/Relu*
T0*,
_output_shapes
:����������
�
)gradients/pi_n/dense_1/Relu_grad/ReluGradReluGrad9gradients/pi_n/dense_2/Tensordot/transpose_grad/transposepi_n/dense_1/Relu*
T0*+
_output_shapes
:���������
�
/gradients/pi_j/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/pi_j/dense_1/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
:
�
4gradients/pi_j/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients/pi_j/dense_1/BiasAdd_grad/BiasAddGrad*^gradients/pi_j/dense_1/Relu_grad/ReluGrad
�
<gradients/pi_j/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/pi_j/dense_1/Relu_grad/ReluGrad5^gradients/pi_j/dense_1/BiasAdd_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/pi_j/dense_1/Relu_grad/ReluGrad*,
_output_shapes
:����������
�
>gradients/pi_j/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/pi_j/dense_1/BiasAdd_grad/BiasAddGrad5^gradients/pi_j/dense_1/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients/pi_j/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
�
/gradients/pi_n/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients/pi_n/dense_1/Relu_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
�
4gradients/pi_n/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients/pi_n/dense_1/BiasAdd_grad/BiasAddGrad*^gradients/pi_n/dense_1/Relu_grad/ReluGrad
�
<gradients/pi_n/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients/pi_n/dense_1/Relu_grad/ReluGrad5^gradients/pi_n/dense_1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients/pi_n/dense_1/Relu_grad/ReluGrad*
T0*+
_output_shapes
:���������
�
>gradients/pi_n/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients/pi_n/dense_1/BiasAdd_grad/BiasAddGrad5^gradients/pi_n/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*B
_class8
64loc:@gradients/pi_n/dense_1/BiasAdd_grad/BiasAddGrad
�
+gradients/pi_j/dense_1/Tensordot_grad/ShapeShapepi_j/dense_1/Tensordot/MatMul*
_output_shapes
:*
out_type0*
T0
�
-gradients/pi_j/dense_1/Tensordot_grad/ReshapeReshape<gradients/pi_j/dense_1/BiasAdd_grad/tuple/control_dependency+gradients/pi_j/dense_1/Tensordot_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
+gradients/pi_n/dense_1/Tensordot_grad/ShapeShapepi_n/dense_1/Tensordot/MatMul*
out_type0*
_output_shapes
:*
T0
�
-gradients/pi_n/dense_1/Tensordot_grad/ReshapeReshape<gradients/pi_n/dense_1/BiasAdd_grad/tuple/control_dependency+gradients/pi_n/dense_1/Tensordot_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
3gradients/pi_j/dense_1/Tensordot/MatMul_grad/MatMulMatMul-gradients/pi_j/dense_1/Tensordot_grad/Reshape pi_j/dense_1/Tensordot/Reshape_1*
T0*
transpose_b(*'
_output_shapes
:��������� *
transpose_a( 
�
5gradients/pi_j/dense_1/Tensordot/MatMul_grad/MatMul_1MatMulpi_j/dense_1/Tensordot/Reshape-gradients/pi_j/dense_1/Tensordot_grad/Reshape*
T0*
transpose_a(*'
_output_shapes
:���������*
transpose_b( 
�
=gradients/pi_j/dense_1/Tensordot/MatMul_grad/tuple/group_depsNoOp4^gradients/pi_j/dense_1/Tensordot/MatMul_grad/MatMul6^gradients/pi_j/dense_1/Tensordot/MatMul_grad/MatMul_1
�
Egradients/pi_j/dense_1/Tensordot/MatMul_grad/tuple/control_dependencyIdentity3gradients/pi_j/dense_1/Tensordot/MatMul_grad/MatMul>^gradients/pi_j/dense_1/Tensordot/MatMul_grad/tuple/group_deps*'
_output_shapes
:��������� *F
_class<
:8loc:@gradients/pi_j/dense_1/Tensordot/MatMul_grad/MatMul*
T0
�
Ggradients/pi_j/dense_1/Tensordot/MatMul_grad/tuple/control_dependency_1Identity5gradients/pi_j/dense_1/Tensordot/MatMul_grad/MatMul_1>^gradients/pi_j/dense_1/Tensordot/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/pi_j/dense_1/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

: 
�
3gradients/pi_n/dense_1/Tensordot/MatMul_grad/MatMulMatMul-gradients/pi_n/dense_1/Tensordot_grad/Reshape pi_n/dense_1/Tensordot/Reshape_1*
transpose_a( *
T0*'
_output_shapes
:��������� *
transpose_b(
�
5gradients/pi_n/dense_1/Tensordot/MatMul_grad/MatMul_1MatMulpi_n/dense_1/Tensordot/Reshape-gradients/pi_n/dense_1/Tensordot_grad/Reshape*
transpose_b( *'
_output_shapes
:���������*
transpose_a(*
T0
�
=gradients/pi_n/dense_1/Tensordot/MatMul_grad/tuple/group_depsNoOp4^gradients/pi_n/dense_1/Tensordot/MatMul_grad/MatMul6^gradients/pi_n/dense_1/Tensordot/MatMul_grad/MatMul_1
�
Egradients/pi_n/dense_1/Tensordot/MatMul_grad/tuple/control_dependencyIdentity3gradients/pi_n/dense_1/Tensordot/MatMul_grad/MatMul>^gradients/pi_n/dense_1/Tensordot/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/pi_n/dense_1/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:��������� *
T0
�
Ggradients/pi_n/dense_1/Tensordot/MatMul_grad/tuple/control_dependency_1Identity5gradients/pi_n/dense_1/Tensordot/MatMul_grad/MatMul_1>^gradients/pi_n/dense_1/Tensordot/MatMul_grad/tuple/group_deps*H
_class>
<:loc:@gradients/pi_n/dense_1/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

: *
T0
�
3gradients/pi_j/dense_1/Tensordot/Reshape_grad/ShapeShape pi_j/dense_1/Tensordot/transpose*
_output_shapes
:*
T0*
out_type0
�
5gradients/pi_j/dense_1/Tensordot/Reshape_grad/ReshapeReshapeEgradients/pi_j/dense_1/Tensordot/MatMul_grad/tuple/control_dependency3gradients/pi_j/dense_1/Tensordot/Reshape_grad/Shape*
Tshape0*
T0*,
_output_shapes
:���������� 
�
5gradients/pi_j/dense_1/Tensordot/Reshape_1_grad/ShapeConst*
dtype0*
valueB"       *
_output_shapes
:
�
7gradients/pi_j/dense_1/Tensordot/Reshape_1_grad/ReshapeReshapeGgradients/pi_j/dense_1/Tensordot/MatMul_grad/tuple/control_dependency_15gradients/pi_j/dense_1/Tensordot/Reshape_1_grad/Shape*
_output_shapes

: *
Tshape0*
T0
�
3gradients/pi_n/dense_1/Tensordot/Reshape_grad/ShapeShape pi_n/dense_1/Tensordot/transpose*
out_type0*
T0*
_output_shapes
:
�
5gradients/pi_n/dense_1/Tensordot/Reshape_grad/ReshapeReshapeEgradients/pi_n/dense_1/Tensordot/MatMul_grad/tuple/control_dependency3gradients/pi_n/dense_1/Tensordot/Reshape_grad/Shape*+
_output_shapes
:��������� *
T0*
Tshape0
�
5gradients/pi_n/dense_1/Tensordot/Reshape_1_grad/ShapeConst*
valueB"       *
dtype0*
_output_shapes
:
�
7gradients/pi_n/dense_1/Tensordot/Reshape_1_grad/ReshapeReshapeGgradients/pi_n/dense_1/Tensordot/MatMul_grad/tuple/control_dependency_15gradients/pi_n/dense_1/Tensordot/Reshape_1_grad/Shape*
_output_shapes

: *
Tshape0*
T0
�
Agradients/pi_j/dense_1/Tensordot/transpose_grad/InvertPermutationInvertPermutationpi_j/dense_1/Tensordot/concat*
_output_shapes
:*
T0
�
9gradients/pi_j/dense_1/Tensordot/transpose_grad/transpose	Transpose5gradients/pi_j/dense_1/Tensordot/Reshape_grad/ReshapeAgradients/pi_j/dense_1/Tensordot/transpose_grad/InvertPermutation*
T0*
Tperm0*,
_output_shapes
:���������� 
�
Cgradients/pi_j/dense_1/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation'pi_j/dense_1/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
;gradients/pi_j/dense_1/Tensordot/transpose_1_grad/transpose	Transpose7gradients/pi_j/dense_1/Tensordot/Reshape_1_grad/ReshapeCgradients/pi_j/dense_1/Tensordot/transpose_1_grad/InvertPermutation*
T0*
Tperm0*
_output_shapes

: 
�
Agradients/pi_n/dense_1/Tensordot/transpose_grad/InvertPermutationInvertPermutationpi_n/dense_1/Tensordot/concat*
_output_shapes
:*
T0
�
9gradients/pi_n/dense_1/Tensordot/transpose_grad/transpose	Transpose5gradients/pi_n/dense_1/Tensordot/Reshape_grad/ReshapeAgradients/pi_n/dense_1/Tensordot/transpose_grad/InvertPermutation*+
_output_shapes
:��������� *
T0*
Tperm0
�
Cgradients/pi_n/dense_1/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation'pi_n/dense_1/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
;gradients/pi_n/dense_1/Tensordot/transpose_1_grad/transpose	Transpose7gradients/pi_n/dense_1/Tensordot/Reshape_1_grad/ReshapeCgradients/pi_n/dense_1/Tensordot/transpose_1_grad/InvertPermutation*
Tperm0*
_output_shapes

: *
T0
�
'gradients/pi_j/dense/Relu_grad/ReluGradReluGrad9gradients/pi_j/dense_1/Tensordot/transpose_grad/transposepi_j/dense/Relu*
T0*,
_output_shapes
:���������� 
�
'gradients/pi_n/dense/Relu_grad/ReluGradReluGrad9gradients/pi_n/dense_1/Tensordot/transpose_grad/transposepi_n/dense/Relu*
T0*+
_output_shapes
:��������� 
�
-gradients/pi_j/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi_j/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
�
2gradients/pi_j/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi_j/dense/BiasAdd_grad/BiasAddGrad(^gradients/pi_j/dense/Relu_grad/ReluGrad
�
:gradients/pi_j/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi_j/dense/Relu_grad/ReluGrad3^gradients/pi_j/dense/BiasAdd_grad/tuple/group_deps*,
_output_shapes
:���������� *:
_class0
.,loc:@gradients/pi_j/dense/Relu_grad/ReluGrad*
T0
�
<gradients/pi_j/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi_j/dense/BiasAdd_grad/BiasAddGrad3^gradients/pi_j/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *@
_class6
42loc:@gradients/pi_j/dense/BiasAdd_grad/BiasAddGrad*
T0
�
-gradients/pi_n/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi_n/dense/Relu_grad/ReluGrad*
data_formatNHWC*
T0*
_output_shapes
: 
�
2gradients/pi_n/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi_n/dense/BiasAdd_grad/BiasAddGrad(^gradients/pi_n/dense/Relu_grad/ReluGrad
�
:gradients/pi_n/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi_n/dense/Relu_grad/ReluGrad3^gradients/pi_n/dense/BiasAdd_grad/tuple/group_deps*+
_output_shapes
:��������� *
T0*:
_class0
.,loc:@gradients/pi_n/dense/Relu_grad/ReluGrad
�
<gradients/pi_n/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi_n/dense/BiasAdd_grad/BiasAddGrad3^gradients/pi_n/dense/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients/pi_n/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
�
)gradients/pi_j/dense/Tensordot_grad/ShapeShapepi_j/dense/Tensordot/MatMul*
out_type0*
_output_shapes
:*
T0
�
+gradients/pi_j/dense/Tensordot_grad/ReshapeReshape:gradients/pi_j/dense/BiasAdd_grad/tuple/control_dependency)gradients/pi_j/dense/Tensordot_grad/Shape*
T0*'
_output_shapes
:��������� *
Tshape0
�
)gradients/pi_n/dense/Tensordot_grad/ShapeShapepi_n/dense/Tensordot/MatMul*
_output_shapes
:*
T0*
out_type0
�
+gradients/pi_n/dense/Tensordot_grad/ReshapeReshape:gradients/pi_n/dense/BiasAdd_grad/tuple/control_dependency)gradients/pi_n/dense/Tensordot_grad/Shape*'
_output_shapes
:��������� *
T0*
Tshape0
�
1gradients/pi_j/dense/Tensordot/MatMul_grad/MatMulMatMul+gradients/pi_j/dense/Tensordot_grad/Reshapepi_j/dense/Tensordot/Reshape_1*
T0*
transpose_a( *'
_output_shapes
:���������*
transpose_b(
�
3gradients/pi_j/dense/Tensordot/MatMul_grad/MatMul_1MatMulpi_j/dense/Tensordot/Reshape+gradients/pi_j/dense/Tensordot_grad/Reshape*
transpose_a(*'
_output_shapes
:��������� *
T0*
transpose_b( 
�
;gradients/pi_j/dense/Tensordot/MatMul_grad/tuple/group_depsNoOp2^gradients/pi_j/dense/Tensordot/MatMul_grad/MatMul4^gradients/pi_j/dense/Tensordot/MatMul_grad/MatMul_1
�
Cgradients/pi_j/dense/Tensordot/MatMul_grad/tuple/control_dependencyIdentity1gradients/pi_j/dense/Tensordot/MatMul_grad/MatMul<^gradients/pi_j/dense/Tensordot/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/pi_j/dense/Tensordot/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Egradients/pi_j/dense/Tensordot/MatMul_grad/tuple/control_dependency_1Identity3gradients/pi_j/dense/Tensordot/MatMul_grad/MatMul_1<^gradients/pi_j/dense/Tensordot/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/pi_j/dense/Tensordot/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
�
1gradients/pi_n/dense/Tensordot/MatMul_grad/MatMulMatMul+gradients/pi_n/dense/Tensordot_grad/Reshapepi_n/dense/Tensordot/Reshape_1*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
3gradients/pi_n/dense/Tensordot/MatMul_grad/MatMul_1MatMulpi_n/dense/Tensordot/Reshape+gradients/pi_n/dense/Tensordot_grad/Reshape*
transpose_a(*'
_output_shapes
:��������� *
transpose_b( *
T0
�
;gradients/pi_n/dense/Tensordot/MatMul_grad/tuple/group_depsNoOp2^gradients/pi_n/dense/Tensordot/MatMul_grad/MatMul4^gradients/pi_n/dense/Tensordot/MatMul_grad/MatMul_1
�
Cgradients/pi_n/dense/Tensordot/MatMul_grad/tuple/control_dependencyIdentity1gradients/pi_n/dense/Tensordot/MatMul_grad/MatMul<^gradients/pi_n/dense/Tensordot/MatMul_grad/tuple/group_deps*D
_class:
86loc:@gradients/pi_n/dense/Tensordot/MatMul_grad/MatMul*
T0*'
_output_shapes
:���������
�
Egradients/pi_n/dense/Tensordot/MatMul_grad/tuple/control_dependency_1Identity3gradients/pi_n/dense/Tensordot/MatMul_grad/MatMul_1<^gradients/pi_n/dense/Tensordot/MatMul_grad/tuple/group_deps*F
_class<
:8loc:@gradients/pi_n/dense/Tensordot/MatMul_grad/MatMul_1*
_output_shapes

: *
T0
�
3gradients/pi_j/dense/Tensordot/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"       
�
5gradients/pi_j/dense/Tensordot/Reshape_1_grad/ReshapeReshapeEgradients/pi_j/dense/Tensordot/MatMul_grad/tuple/control_dependency_13gradients/pi_j/dense/Tensordot/Reshape_1_grad/Shape*
_output_shapes

: *
T0*
Tshape0
�
3gradients/pi_n/dense/Tensordot/Reshape_1_grad/ShapeConst*
dtype0*
valueB"       *
_output_shapes
:
�
5gradients/pi_n/dense/Tensordot/Reshape_1_grad/ReshapeReshapeEgradients/pi_n/dense/Tensordot/MatMul_grad/tuple/control_dependency_13gradients/pi_n/dense/Tensordot/Reshape_1_grad/Shape*
Tshape0*
_output_shapes

: *
T0
�
Agradients/pi_j/dense/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation%pi_j/dense/Tensordot/transpose_1/perm*
_output_shapes
:*
T0
�
9gradients/pi_j/dense/Tensordot/transpose_1_grad/transpose	Transpose5gradients/pi_j/dense/Tensordot/Reshape_1_grad/ReshapeAgradients/pi_j/dense/Tensordot/transpose_1_grad/InvertPermutation*
_output_shapes

: *
T0*
Tperm0
�
Agradients/pi_n/dense/Tensordot/transpose_1_grad/InvertPermutationInvertPermutation%pi_n/dense/Tensordot/transpose_1/perm*
T0*
_output_shapes
:
�
9gradients/pi_n/dense/Tensordot/transpose_1_grad/transpose	Transpose5gradients/pi_n/dense/Tensordot/Reshape_1_grad/ReshapeAgradients/pi_n/dense/Tensordot/transpose_1_grad/InvertPermutation*
_output_shapes

: *
Tperm0*
T0
�
beta1_power/initial_valueConst*
valueB
 *fff?*
_output_shapes
: *
dtype0*"
_class
loc:@pi_j/dense/bias
�
beta1_power
VariableV2*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
shape: *
	container *
shared_name *
dtype0
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
use_locking(
n
beta1_power/readIdentitybeta1_power*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
T0
�
beta2_power/initial_valueConst*
valueB
 *w�?*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
dtype0
�
beta2_power
VariableV2*
shared_name *
_output_shapes
: *
	container *
shape: *"
_class
loc:@pi_j/dense/bias*
dtype0
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
validate_shape(*"
_class
loc:@pi_j/dense/bias*
T0*
_output_shapes
: 
n
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*"
_class
loc:@pi_j/dense/bias
�
(pi_j/dense/kernel/Adam/Initializer/zerosConst*$
_class
loc:@pi_j/dense/kernel*
_output_shapes

: *
dtype0*
valueB *    
�
pi_j/dense/kernel/Adam
VariableV2*
shared_name *
	container *
shape
: *
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel*
dtype0
�
pi_j/dense/kernel/Adam/AssignAssignpi_j/dense/kernel/Adam(pi_j/dense/kernel/Adam/Initializer/zeros*
_output_shapes

: *
use_locking(*$
_class
loc:@pi_j/dense/kernel*
validate_shape(*
T0
�
pi_j/dense/kernel/Adam/readIdentitypi_j/dense/kernel/Adam*
T0*
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel
�
*pi_j/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

: *
valueB *    *
dtype0*$
_class
loc:@pi_j/dense/kernel
�
pi_j/dense/kernel/Adam_1
VariableV2*
shape
: *
	container *
dtype0*
shared_name *$
_class
loc:@pi_j/dense/kernel*
_output_shapes

: 
�
pi_j/dense/kernel/Adam_1/AssignAssignpi_j/dense/kernel/Adam_1*pi_j/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

: *
use_locking(*$
_class
loc:@pi_j/dense/kernel*
T0*
validate_shape(
�
pi_j/dense/kernel/Adam_1/readIdentitypi_j/dense/kernel/Adam_1*
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel*
T0
�
&pi_j/dense/bias/Adam/Initializer/zerosConst*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
dtype0*
valueB *    
�
pi_j/dense/bias/Adam
VariableV2*
_output_shapes
: *
	container *
dtype0*
shared_name *
shape: *"
_class
loc:@pi_j/dense/bias
�
pi_j/dense/bias/Adam/AssignAssignpi_j/dense/bias/Adam&pi_j/dense/bias/Adam/Initializer/zeros*
_output_shapes
: *
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi_j/dense/bias
�
pi_j/dense/bias/Adam/readIdentitypi_j/dense/bias/Adam*
_output_shapes
: *
T0*"
_class
loc:@pi_j/dense/bias
�
(pi_j/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
: *
dtype0*
valueB *    *"
_class
loc:@pi_j/dense/bias
�
pi_j/dense/bias/Adam_1
VariableV2*
shape: *"
_class
loc:@pi_j/dense/bias*
	container *
shared_name *
dtype0*
_output_shapes
: 
�
pi_j/dense/bias/Adam_1/AssignAssignpi_j/dense/bias/Adam_1(pi_j/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@pi_j/dense/bias
�
pi_j/dense/bias/Adam_1/readIdentitypi_j/dense/bias/Adam_1*
T0*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias
�
*pi_j/dense_1/kernel/Adam/Initializer/zerosConst*
valueB *    *&
_class
loc:@pi_j/dense_1/kernel*
dtype0*
_output_shapes

: 
�
pi_j/dense_1/kernel/Adam
VariableV2*&
_class
loc:@pi_j/dense_1/kernel*
	container *
dtype0*
shared_name *
_output_shapes

: *
shape
: 
�
pi_j/dense_1/kernel/Adam/AssignAssignpi_j/dense_1/kernel/Adam*pi_j/dense_1/kernel/Adam/Initializer/zeros*
validate_shape(*&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: *
use_locking(*
T0
�
pi_j/dense_1/kernel/Adam/readIdentitypi_j/dense_1/kernel/Adam*
T0*&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: 
�
,pi_j/dense_1/kernel/Adam_1/Initializer/zerosConst*
valueB *    *&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: *
dtype0
�
pi_j/dense_1/kernel/Adam_1
VariableV2*
_output_shapes

: *
shared_name *&
_class
loc:@pi_j/dense_1/kernel*
	container *
shape
: *
dtype0
�
!pi_j/dense_1/kernel/Adam_1/AssignAssignpi_j/dense_1/kernel/Adam_1,pi_j/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes

: *
T0*&
_class
loc:@pi_j/dense_1/kernel
�
pi_j/dense_1/kernel/Adam_1/readIdentitypi_j/dense_1/kernel/Adam_1*&
_class
loc:@pi_j/dense_1/kernel*
T0*
_output_shapes

: 
�
(pi_j/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*$
_class
loc:@pi_j/dense_1/bias*
valueB*    *
_output_shapes
:
�
pi_j/dense_1/bias/Adam
VariableV2*
dtype0*
	container *$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:*
shape:*
shared_name 
�
pi_j/dense_1/bias/Adam/AssignAssignpi_j/dense_1/bias/Adam(pi_j/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:
�
pi_j/dense_1/bias/Adam/readIdentitypi_j/dense_1/bias/Adam*
T0*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:
�
*pi_j/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:
�
pi_j/dense_1/bias/Adam_1
VariableV2*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:*
dtype0*
shape:*
	container *
shared_name 
�
pi_j/dense_1/bias/Adam_1/AssignAssignpi_j/dense_1/bias/Adam_1*pi_j/dense_1/bias/Adam_1/Initializer/zeros*
T0*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
pi_j/dense_1/bias/Adam_1/readIdentitypi_j/dense_1/bias/Adam_1*$
_class
loc:@pi_j/dense_1/bias*
T0*
_output_shapes
:
�
*pi_j/dense_2/kernel/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes

:*&
_class
loc:@pi_j/dense_2/kernel
�
pi_j/dense_2/kernel/Adam
VariableV2*&
_class
loc:@pi_j/dense_2/kernel*
dtype0*
	container *
_output_shapes

:*
shared_name *
shape
:
�
pi_j/dense_2/kernel/Adam/AssignAssignpi_j/dense_2/kernel/Adam*pi_j/dense_2/kernel/Adam/Initializer/zeros*&
_class
loc:@pi_j/dense_2/kernel*
use_locking(*
_output_shapes

:*
T0*
validate_shape(
�
pi_j/dense_2/kernel/Adam/readIdentitypi_j/dense_2/kernel/Adam*&
_class
loc:@pi_j/dense_2/kernel*
T0*
_output_shapes

:
�
,pi_j/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes

:*
dtype0*&
_class
loc:@pi_j/dense_2/kernel
�
pi_j/dense_2/kernel/Adam_1
VariableV2*
shape
:*
	container *
dtype0*&
_class
loc:@pi_j/dense_2/kernel*
_output_shapes

:*
shared_name 
�
!pi_j/dense_2/kernel/Adam_1/AssignAssignpi_j/dense_2/kernel/Adam_1,pi_j/dense_2/kernel/Adam_1/Initializer/zeros*&
_class
loc:@pi_j/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes

:
�
pi_j/dense_2/kernel/Adam_1/readIdentitypi_j/dense_2/kernel/Adam_1*
T0*&
_class
loc:@pi_j/dense_2/kernel*
_output_shapes

:
�
(pi_j/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *$
_class
loc:@pi_j/dense_2/bias
�
pi_j/dense_2/bias/Adam
VariableV2*
shared_name *
	container *$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:*
shape:*
dtype0
�
pi_j/dense_2/bias/Adam/AssignAssignpi_j/dense_2/bias/Adam(pi_j/dense_2/bias/Adam/Initializer/zeros*
T0*
validate_shape(*$
_class
loc:@pi_j/dense_2/bias*
use_locking(*
_output_shapes
:
�
pi_j/dense_2/bias/Adam/readIdentitypi_j/dense_2/bias/Adam*
T0*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:
�
*pi_j/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*$
_class
loc:@pi_j/dense_2/bias*
dtype0*
valueB*    
�
pi_j/dense_2/bias/Adam_1
VariableV2*$
_class
loc:@pi_j/dense_2/bias*
shape:*
	container *
_output_shapes
:*
dtype0*
shared_name 
�
pi_j/dense_2/bias/Adam_1/AssignAssignpi_j/dense_2/bias/Adam_1*pi_j/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:
�
pi_j/dense_2/bias/Adam_1/readIdentitypi_j/dense_2/bias/Adam_1*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:*
T0
�
*pi_j/dense_3/kernel/Adam/Initializer/zerosConst*
dtype0*&
_class
loc:@pi_j/dense_3/kernel*
valueB*    *
_output_shapes

:
�
pi_j/dense_3/kernel/Adam
VariableV2*
	container *
dtype0*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel*
shape
:*
shared_name 
�
pi_j/dense_3/kernel/Adam/AssignAssignpi_j/dense_3/kernel/Adam*pi_j/dense_3/kernel/Adam/Initializer/zeros*
T0*&
_class
loc:@pi_j/dense_3/kernel*
use_locking(*
validate_shape(*
_output_shapes

:
�
pi_j/dense_3/kernel/Adam/readIdentitypi_j/dense_3/kernel/Adam*
T0*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:
�
,pi_j/dense_3/kernel/Adam_1/Initializer/zerosConst*&
_class
loc:@pi_j/dense_3/kernel*
valueB*    *
dtype0*
_output_shapes

:
�
pi_j/dense_3/kernel/Adam_1
VariableV2*
shared_name *
shape
:*&
_class
loc:@pi_j/dense_3/kernel*
	container *
_output_shapes

:*
dtype0
�
!pi_j/dense_3/kernel/Adam_1/AssignAssignpi_j/dense_3/kernel/Adam_1,pi_j/dense_3/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:*
use_locking(
�
pi_j/dense_3/kernel/Adam_1/readIdentitypi_j/dense_3/kernel/Adam_1*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel*
T0
�
(pi_j/dense_3/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
dtype0*$
_class
loc:@pi_j/dense_3/bias*
valueB*    
�
pi_j/dense_3/bias/Adam
VariableV2*
	container *
_output_shapes
:*
shape:*
dtype0*
shared_name *$
_class
loc:@pi_j/dense_3/bias
�
pi_j/dense_3/bias/Adam/AssignAssignpi_j/dense_3/bias/Adam(pi_j/dense_3/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias*
T0*
use_locking(
�
pi_j/dense_3/bias/Adam/readIdentitypi_j/dense_3/bias/Adam*
_output_shapes
:*
T0*$
_class
loc:@pi_j/dense_3/bias
�
*pi_j/dense_3/bias/Adam_1/Initializer/zerosConst*$
_class
loc:@pi_j/dense_3/bias*
dtype0*
_output_shapes
:*
valueB*    
�
pi_j/dense_3/bias/Adam_1
VariableV2*
_output_shapes
:*
shared_name *
shape:*
	container *
dtype0*$
_class
loc:@pi_j/dense_3/bias
�
pi_j/dense_3/bias/Adam_1/AssignAssignpi_j/dense_3/bias/Adam_1*pi_j/dense_3/bias/Adam_1/Initializer/zeros*$
_class
loc:@pi_j/dense_3/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
�
pi_j/dense_3/bias/Adam_1/readIdentitypi_j/dense_3/bias/Adam_1*$
_class
loc:@pi_j/dense_3/bias*
_output_shapes
:*
T0
�
(pi_n/dense/kernel/Adam/Initializer/zerosConst*
valueB *    *$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: *
dtype0
�
pi_n/dense/kernel/Adam
VariableV2*
shared_name *
shape
: *
_output_shapes

: *
	container *$
_class
loc:@pi_n/dense/kernel*
dtype0
�
pi_n/dense/kernel/Adam/AssignAssignpi_n/dense/kernel/Adam(pi_n/dense/kernel/Adam/Initializer/zeros*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel*
validate_shape(*
T0*
use_locking(
�
pi_n/dense/kernel/Adam/readIdentitypi_n/dense/kernel/Adam*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel*
T0
�
*pi_n/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

: *
dtype0*
valueB *    *$
_class
loc:@pi_n/dense/kernel
�
pi_n/dense/kernel/Adam_1
VariableV2*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: *
	container *
shape
: *
shared_name *
dtype0
�
pi_n/dense/kernel/Adam_1/AssignAssignpi_n/dense/kernel/Adam_1*pi_n/dense/kernel/Adam_1/Initializer/zeros*$
_class
loc:@pi_n/dense/kernel*
T0*
_output_shapes

: *
validate_shape(*
use_locking(
�
pi_n/dense/kernel/Adam_1/readIdentitypi_n/dense/kernel/Adam_1*
T0*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: 
�
&pi_n/dense/bias/Adam/Initializer/zerosConst*
dtype0*"
_class
loc:@pi_n/dense/bias*
_output_shapes
: *
valueB *    
�
pi_n/dense/bias/Adam
VariableV2*"
_class
loc:@pi_n/dense/bias*
shape: *
	container *
dtype0*
shared_name *
_output_shapes
: 
�
pi_n/dense/bias/Adam/AssignAssignpi_n/dense/bias/Adam&pi_n/dense/bias/Adam/Initializer/zeros*"
_class
loc:@pi_n/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
�
pi_n/dense/bias/Adam/readIdentitypi_n/dense/bias/Adam*
T0*"
_class
loc:@pi_n/dense/bias*
_output_shapes
: 
�
(pi_n/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB *    *"
_class
loc:@pi_n/dense/bias*
_output_shapes
: 
�
pi_n/dense/bias/Adam_1
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes
: *"
_class
loc:@pi_n/dense/bias*
shape: 
�
pi_n/dense/bias/Adam_1/AssignAssignpi_n/dense/bias/Adam_1(pi_n/dense/bias/Adam_1/Initializer/zeros*"
_class
loc:@pi_n/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
�
pi_n/dense/bias/Adam_1/readIdentitypi_n/dense/bias/Adam_1*
_output_shapes
: *
T0*"
_class
loc:@pi_n/dense/bias
�
*pi_n/dense_1/kernel/Adam/Initializer/zerosConst*
valueB *    *
dtype0*&
_class
loc:@pi_n/dense_1/kernel*
_output_shapes

: 
�
pi_n/dense_1/kernel/Adam
VariableV2*
_output_shapes

: *
	container *
shape
: *
dtype0*&
_class
loc:@pi_n/dense_1/kernel*
shared_name 
�
pi_n/dense_1/kernel/Adam/AssignAssignpi_n/dense_1/kernel/Adam*pi_n/dense_1/kernel/Adam/Initializer/zeros*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
validate_shape(*
use_locking(*
T0
�
pi_n/dense_1/kernel/Adam/readIdentitypi_n/dense_1/kernel/Adam*
T0*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel
�
,pi_n/dense_1/kernel/Adam_1/Initializer/zerosConst*
valueB *    *
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
dtype0
�
pi_n/dense_1/kernel/Adam_1
VariableV2*&
_class
loc:@pi_n/dense_1/kernel*
	container *
dtype0*
shared_name *
_output_shapes

: *
shape
: 
�
!pi_n/dense_1/kernel/Adam_1/AssignAssignpi_n/dense_1/kernel/Adam_1,pi_n/dense_1/kernel/Adam_1/Initializer/zeros*&
_class
loc:@pi_n/dense_1/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
�
pi_n/dense_1/kernel/Adam_1/readIdentitypi_n/dense_1/kernel/Adam_1*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
T0
�
(pi_n/dense_1/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *$
_class
loc:@pi_n/dense_1/bias*
dtype0
�
pi_n/dense_1/bias/Adam
VariableV2*
shared_name *
shape:*
	container *
dtype0*
_output_shapes
:*$
_class
loc:@pi_n/dense_1/bias
�
pi_n/dense_1/bias/Adam/AssignAssignpi_n/dense_1/bias/Adam(pi_n/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*$
_class
loc:@pi_n/dense_1/bias
�
pi_n/dense_1/bias/Adam/readIdentitypi_n/dense_1/bias/Adam*
T0*
_output_shapes
:*$
_class
loc:@pi_n/dense_1/bias
�
*pi_n/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
valueB*    *$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:
�
pi_n/dense_1/bias/Adam_1
VariableV2*
shared_name *
_output_shapes
:*
	container *
dtype0*$
_class
loc:@pi_n/dense_1/bias*
shape:
�
pi_n/dense_1/bias/Adam_1/AssignAssignpi_n/dense_1/bias/Adam_1*pi_n/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_n/dense_1/bias*
T0
�
pi_n/dense_1/bias/Adam_1/readIdentitypi_n/dense_1/bias/Adam_1*$
_class
loc:@pi_n/dense_1/bias*
T0*
_output_shapes
:
�
*pi_n/dense_2/kernel/Adam/Initializer/zerosConst*&
_class
loc:@pi_n/dense_2/kernel*
dtype0*
_output_shapes

:*
valueB*    
�
pi_n/dense_2/kernel/Adam
VariableV2*
	container *
shared_name *
dtype0*
_output_shapes

:*&
_class
loc:@pi_n/dense_2/kernel*
shape
:
�
pi_n/dense_2/kernel/Adam/AssignAssignpi_n/dense_2/kernel/Adam*pi_n/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:*
use_locking(*
T0
�
pi_n/dense_2/kernel/Adam/readIdentitypi_n/dense_2/kernel/Adam*
_output_shapes

:*&
_class
loc:@pi_n/dense_2/kernel*
T0
�
,pi_n/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:*
dtype0*
valueB*    *&
_class
loc:@pi_n/dense_2/kernel
�
pi_n/dense_2/kernel/Adam_1
VariableV2*
	container *
_output_shapes

:*
shared_name *&
_class
loc:@pi_n/dense_2/kernel*
shape
:*
dtype0
�
!pi_n/dense_2/kernel/Adam_1/AssignAssignpi_n/dense_2/kernel/Adam_1,pi_n/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:
�
pi_n/dense_2/kernel/Adam_1/readIdentitypi_n/dense_2/kernel/Adam_1*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_2/kernel
�
(pi_n/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *$
_class
loc:@pi_n/dense_2/bias*
dtype0
�
pi_n/dense_2/bias/Adam
VariableV2*
	container *
shape:*
_output_shapes
:*
dtype0*$
_class
loc:@pi_n/dense_2/bias*
shared_name 
�
pi_n/dense_2/bias/Adam/AssignAssignpi_n/dense_2/bias/Adam(pi_n/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
use_locking(*
T0
�
pi_n/dense_2/bias/Adam/readIdentitypi_n/dense_2/bias/Adam*
T0*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias
�
*pi_n/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
valueB*    
�
pi_n/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
	container *
shared_name *$
_class
loc:@pi_n/dense_2/bias*
shape:
�
pi_n/dense_2/bias/Adam_1/AssignAssignpi_n/dense_2/bias/Adam_1*pi_n/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_n/dense_2/bias
�
pi_n/dense_2/bias/Adam_1/readIdentitypi_n/dense_2/bias/Adam_1*
T0*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias
�
*pi_n/dense_3/kernel/Adam/Initializer/zerosConst*&
_class
loc:@pi_n/dense_3/kernel*
dtype0*
_output_shapes

:*
valueB*    
�
pi_n/dense_3/kernel/Adam
VariableV2*
_output_shapes

:*
	container *
shared_name *
shape
:*
dtype0*&
_class
loc:@pi_n/dense_3/kernel
�
pi_n/dense_3/kernel/Adam/AssignAssignpi_n/dense_3/kernel/Adam*pi_n/dense_3/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*&
_class
loc:@pi_n/dense_3/kernel*
_output_shapes

:*
use_locking(
�
pi_n/dense_3/kernel/Adam/readIdentitypi_n/dense_3/kernel/Adam*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel
�
,pi_n/dense_3/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:*
dtype0*
valueB*    *&
_class
loc:@pi_n/dense_3/kernel
�
pi_n/dense_3/kernel/Adam_1
VariableV2*
_output_shapes

:*
dtype0*
shared_name *&
_class
loc:@pi_n/dense_3/kernel*
	container *
shape
:
�
!pi_n/dense_3/kernel/Adam_1/AssignAssignpi_n/dense_3/kernel/Adam_1,pi_n/dense_3/kernel/Adam_1/Initializer/zeros*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel*
validate_shape(*
use_locking(
�
pi_n/dense_3/kernel/Adam_1/readIdentitypi_n/dense_3/kernel/Adam_1*
_output_shapes

:*
T0*&
_class
loc:@pi_n/dense_3/kernel
�
(pi_n/dense_3/bias/Adam/Initializer/zerosConst*$
_class
loc:@pi_n/dense_3/bias*
dtype0*
valueB*    *
_output_shapes
:
�
pi_n/dense_3/bias/Adam
VariableV2*
shape:*
	container *
_output_shapes
:*
dtype0*$
_class
loc:@pi_n/dense_3/bias*
shared_name 
�
pi_n/dense_3/bias/Adam/AssignAssignpi_n/dense_3/bias/Adam(pi_n/dense_3/bias/Adam/Initializer/zeros*
T0*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
use_locking(
�
pi_n/dense_3/bias/Adam/readIdentitypi_n/dense_3/bias/Adam*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_3/bias
�
*pi_n/dense_3/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *$
_class
loc:@pi_n/dense_3/bias
�
pi_n/dense_3/bias/Adam_1
VariableV2*
shared_name *
_output_shapes
:*
	container *$
_class
loc:@pi_n/dense_3/bias*
dtype0*
shape:
�
pi_n/dense_3/bias/Adam_1/AssignAssignpi_n/dense_3/bias/Adam_1*pi_n/dense_3/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
T0
�
pi_n/dense_3/bias/Adam_1/readIdentitypi_n/dense_3/bias/Adam_1*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_3/bias
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *RI�9*
dtype0
O

Adam/beta1Const*
valueB
 *fff?*
_output_shapes
: *
dtype0
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
_output_shapes
: *
dtype0*
valueB
 *w�+2
�
'Adam/update_pi_j/dense/kernel/ApplyAdam	ApplyAdampi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/pi_j/dense/Tensordot/transpose_1_grad/transpose*
T0*
_output_shapes

: *
use_locking( *$
_class
loc:@pi_j/dense/kernel*
use_nesterov( 
�
%Adam/update_pi_j/dense/bias/ApplyAdam	ApplyAdampi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/pi_j/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
use_nesterov( *"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
use_locking( 
�
)Adam/update_pi_j/dense_1/kernel/ApplyAdam	ApplyAdampi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/pi_j/dense_1/Tensordot/transpose_1_grad/transpose*
use_locking( *
T0*
use_nesterov( *&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: 
�
'Adam/update_pi_j/dense_1/bias/ApplyAdam	ApplyAdampi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/pi_j/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:*
use_locking( *$
_class
loc:@pi_j/dense_1/bias*
use_nesterov( 
�
)Adam/update_pi_j/dense_2/kernel/ApplyAdam	ApplyAdampi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/pi_j/dense_2/Tensordot/transpose_1_grad/transpose*
_output_shapes

:*
use_nesterov( *&
_class
loc:@pi_j/dense_2/kernel*
T0*
use_locking( 
�
'Adam/update_pi_j/dense_2/bias/ApplyAdam	ApplyAdampi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/pi_j/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *$
_class
loc:@pi_j/dense_2/bias*
T0*
use_nesterov( *
_output_shapes
:
�
)Adam/update_pi_j/dense_3/kernel/ApplyAdam	ApplyAdampi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/pi_j/dense_3/Tensordot/transpose_1_grad/transpose*
use_nesterov( *
T0*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:*
use_locking( 
�
'Adam/update_pi_j/dense_3/bias/ApplyAdam	ApplyAdampi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/pi_j/dense_3/BiasAdd_grad/tuple/control_dependency_1*$
_class
loc:@pi_j/dense_3/bias*
_output_shapes
:*
T0*
use_nesterov( *
use_locking( 
�
'Adam/update_pi_n/dense/kernel/ApplyAdam	ApplyAdampi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/pi_n/dense/Tensordot/transpose_1_grad/transpose*$
_class
loc:@pi_n/dense/kernel*
use_locking( *
T0*
use_nesterov( *
_output_shapes

: 
�
%Adam/update_pi_n/dense/bias/ApplyAdam	ApplyAdampi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon<gradients/pi_n/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
: *
use_locking( *
use_nesterov( *"
_class
loc:@pi_n/dense/bias*
T0
�
)Adam/update_pi_n/dense_1/kernel/ApplyAdam	ApplyAdampi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/pi_n/dense_1/Tensordot/transpose_1_grad/transpose*
T0*
_output_shapes

: *
use_nesterov( *&
_class
loc:@pi_n/dense_1/kernel*
use_locking( 
�
'Adam/update_pi_n/dense_1/bias/ApplyAdam	ApplyAdampi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/pi_n/dense_1/BiasAdd_grad/tuple/control_dependency_1*$
_class
loc:@pi_n/dense_1/bias*
use_locking( *
_output_shapes
:*
use_nesterov( *
T0
�
)Adam/update_pi_n/dense_2/kernel/ApplyAdam	ApplyAdampi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/pi_n/dense_2/Tensordot/transpose_1_grad/transpose*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:*
T0*
use_nesterov( *
use_locking( 
�
'Adam/update_pi_n/dense_2/bias/ApplyAdam	ApplyAdampi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/pi_n/dense_2/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes
:*
use_locking( *
use_nesterov( *$
_class
loc:@pi_n/dense_2/bias*
T0
�
)Adam/update_pi_n/dense_3/kernel/ApplyAdam	ApplyAdampi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon;gradients/pi_n/dense_3/Tensordot/transpose_1_grad/transpose*&
_class
loc:@pi_n/dense_3/kernel*
use_nesterov( *
_output_shapes

:*
T0*
use_locking( 
�
'Adam/update_pi_n/dense_3/bias/ApplyAdam	ApplyAdampi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon>gradients/pi_n/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
_output_shapes
:*$
_class
loc:@pi_n/dense_3/bias*
T0
�
Adam/mulMulbeta1_power/read
Adam/beta1&^Adam/update_pi_j/dense/bias/ApplyAdam(^Adam/update_pi_j/dense/kernel/ApplyAdam(^Adam/update_pi_j/dense_1/bias/ApplyAdam*^Adam/update_pi_j/dense_1/kernel/ApplyAdam(^Adam/update_pi_j/dense_2/bias/ApplyAdam*^Adam/update_pi_j/dense_2/kernel/ApplyAdam(^Adam/update_pi_j/dense_3/bias/ApplyAdam*^Adam/update_pi_j/dense_3/kernel/ApplyAdam&^Adam/update_pi_n/dense/bias/ApplyAdam(^Adam/update_pi_n/dense/kernel/ApplyAdam(^Adam/update_pi_n/dense_1/bias/ApplyAdam*^Adam/update_pi_n/dense_1/kernel/ApplyAdam(^Adam/update_pi_n/dense_2/bias/ApplyAdam*^Adam/update_pi_n/dense_2/kernel/ApplyAdam(^Adam/update_pi_n/dense_3/bias/ApplyAdam*^Adam/update_pi_n/dense_3/kernel/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@pi_j/dense/bias
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
validate_shape(
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2&^Adam/update_pi_j/dense/bias/ApplyAdam(^Adam/update_pi_j/dense/kernel/ApplyAdam(^Adam/update_pi_j/dense_1/bias/ApplyAdam*^Adam/update_pi_j/dense_1/kernel/ApplyAdam(^Adam/update_pi_j/dense_2/bias/ApplyAdam*^Adam/update_pi_j/dense_2/kernel/ApplyAdam(^Adam/update_pi_j/dense_3/bias/ApplyAdam*^Adam/update_pi_j/dense_3/kernel/ApplyAdam&^Adam/update_pi_n/dense/bias/ApplyAdam(^Adam/update_pi_n/dense/kernel/ApplyAdam(^Adam/update_pi_n/dense_1/bias/ApplyAdam*^Adam/update_pi_n/dense_1/kernel/ApplyAdam(^Adam/update_pi_n/dense_2/bias/ApplyAdam*^Adam/update_pi_n/dense_2/kernel/ApplyAdam(^Adam/update_pi_n/dense_3/bias/ApplyAdam*^Adam/update_pi_n/dense_3/kernel/ApplyAdam*
T0*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*"
_class
loc:@pi_j/dense/bias*
T0*
use_locking( *
_output_shapes
: *
validate_shape(
�
AdamNoOp^Adam/Assign^Adam/Assign_1&^Adam/update_pi_j/dense/bias/ApplyAdam(^Adam/update_pi_j/dense/kernel/ApplyAdam(^Adam/update_pi_j/dense_1/bias/ApplyAdam*^Adam/update_pi_j/dense_1/kernel/ApplyAdam(^Adam/update_pi_j/dense_2/bias/ApplyAdam*^Adam/update_pi_j/dense_2/kernel/ApplyAdam(^Adam/update_pi_j/dense_3/bias/ApplyAdam*^Adam/update_pi_j/dense_3/kernel/ApplyAdam&^Adam/update_pi_n/dense/bias/ApplyAdam(^Adam/update_pi_n/dense/kernel/ApplyAdam(^Adam/update_pi_n/dense_1/bias/ApplyAdam*^Adam/update_pi_n/dense_1/kernel/ApplyAdam(^Adam/update_pi_n/dense_2/bias/ApplyAdam*^Adam/update_pi_n/dense_2/kernel/ApplyAdam(^Adam/update_pi_n/dense_3/bias/ApplyAdam*^Adam/update_pi_n/dense_3/kernel/ApplyAdam
T
gradients_1/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
gradients_1/grad_ys_0Const*
dtype0*
valueB
 *  �?*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *

index_type0*
T0
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
�
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
T0*
_output_shapes
:*
Tshape0
`
gradients_1/Mean_1_grad/ShapeShapepow*
_output_shapes
:*
T0*
out_type0
�
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*#
_output_shapes
:���������*
T0
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
_output_shapes
:*
T0*
out_type0
b
gradients_1/Mean_1_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
g
gradients_1/Mean_1_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
�
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
�
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
�
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
�
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
_output_shapes
: *
T0
�
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*#
_output_shapes
:���������*
T0
_
gradients_1/pow_grad/ShapeShapesub_2*
_output_shapes
:*
out_type0*
T0
_
gradients_1/pow_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*#
_output_shapes
:���������*
T0
_
gradients_1/pow_grad/sub/yConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_1/pow_grad/PowPowsub_2gradients_1/pow_grad/sub*
T0*#
_output_shapes
:���������
�
gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*#
_output_shapes
:���������*
T0
�
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
�
gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*
T0*#
_output_shapes
:���������*
Tshape0
c
gradients_1/pow_grad/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
|
gradients_1/pow_grad/GreaterGreatersub_2gradients_1/pow_grad/Greater/y*
T0*#
_output_shapes
:���������
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_2*
_output_shapes
:*
out_type0*
T0
i
$gradients_1/pow_grad/ones_like/ConstConst*
valueB
 *  �?*
_output_shapes
: *
dtype0
�
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*

index_type0*
T0*#
_output_shapes
:���������
�
gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_2gradients_1/pow_grad/ones_like*#
_output_shapes
:���������*
T0
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*
T0*#
_output_shapes
:���������
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_2*#
_output_shapes
:���������*
T0
�
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*#
_output_shapes
:���������*
T0
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*#
_output_shapes
:���������*
T0
�
gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*
T0*#
_output_shapes
:���������
�
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
�
gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
�
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*#
_output_shapes
:���������
�
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*
_output_shapes
: *
T0*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1
i
gradients_1/sub_2_grad/ShapeShapePlaceholder_8*
T0*
_output_shapes
:*
out_type0
g
gradients_1/sub_2_grad/Shape_1Shape	v/Squeeze*
_output_shapes
:*
out_type0*
T0
�
,gradients_1/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_2_grad/Shapegradients_1/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/sub_2_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_2_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients_1/sub_2_grad/ReshapeReshapegradients_1/sub_2_grad/Sumgradients_1/sub_2_grad/Shape*
Tshape0*
T0*#
_output_shapes
:���������
�
gradients_1/sub_2_grad/Sum_1Sum-gradients_1/pow_grad/tuple/control_dependency.gradients_1/sub_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
b
gradients_1/sub_2_grad/NegNeggradients_1/sub_2_grad/Sum_1*
T0*
_output_shapes
:
�
 gradients_1/sub_2_grad/Reshape_1Reshapegradients_1/sub_2_grad/Neggradients_1/sub_2_grad/Shape_1*
T0*#
_output_shapes
:���������*
Tshape0
s
'gradients_1/sub_2_grad/tuple/group_depsNoOp^gradients_1/sub_2_grad/Reshape!^gradients_1/sub_2_grad/Reshape_1
�
/gradients_1/sub_2_grad/tuple/control_dependencyIdentitygradients_1/sub_2_grad/Reshape(^gradients_1/sub_2_grad/tuple/group_deps*#
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients_1/sub_2_grad/Reshape
�
1gradients_1/sub_2_grad/tuple/control_dependency_1Identity gradients_1/sub_2_grad/Reshape_1(^gradients_1/sub_2_grad/tuple/group_deps*#
_output_shapes
:���������*3
_class)
'%loc:@gradients_1/sub_2_grad/Reshape_1*
T0
q
 gradients_1/v/Squeeze_grad/ShapeShapev/dense_3/BiasAdd*
T0*
_output_shapes
:*
out_type0
�
"gradients_1/v/Squeeze_grad/ReshapeReshape1gradients_1/sub_2_grad/tuple/control_dependency_1 gradients_1/v/Squeeze_grad/Shape*
Tshape0*
T0*'
_output_shapes
:���������
�
.gradients_1/v/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients_1/v/Squeeze_grad/Reshape*
_output_shapes
:*
data_formatNHWC*
T0
�
3gradients_1/v/dense_3/BiasAdd_grad/tuple/group_depsNoOp#^gradients_1/v/Squeeze_grad/Reshape/^gradients_1/v/dense_3/BiasAdd_grad/BiasAddGrad
�
;gradients_1/v/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity"gradients_1/v/Squeeze_grad/Reshape4^gradients_1/v/dense_3/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/v/Squeeze_grad/Reshape*'
_output_shapes
:���������
�
=gradients_1/v/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_3/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_3/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/v/dense_3/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
(gradients_1/v/dense_3/MatMul_grad/MatMulMatMul;gradients_1/v/dense_3/BiasAdd_grad/tuple/control_dependencyv/dense_3/kernel/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
*gradients_1/v/dense_3/MatMul_grad/MatMul_1MatMulv/dense_2/Relu;gradients_1/v/dense_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:*
transpose_a(*
T0
�
2gradients_1/v/dense_3/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_3/MatMul_grad/MatMul+^gradients_1/v/dense_3/MatMul_grad/MatMul_1
�
:gradients_1/v/dense_3/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_3/MatMul_grad/MatMul3^gradients_1/v/dense_3/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*;
_class1
/-loc:@gradients_1/v/dense_3/MatMul_grad/MatMul*
T0
�
<gradients_1/v/dense_3/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_3/MatMul_grad/MatMul_13^gradients_1/v/dense_3/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*=
_class3
1/loc:@gradients_1/v/dense_3/MatMul_grad/MatMul_1
�
(gradients_1/v/dense_2/Relu_grad/ReluGradReluGrad:gradients_1/v/dense_3/MatMul_grad/tuple/control_dependencyv/dense_2/Relu*'
_output_shapes
:���������*
T0
�
.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_2/Relu_grad/ReluGrad*
T0*
_output_shapes
:*
data_formatNHWC
�
3gradients_1/v/dense_2/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_2/Relu_grad/ReluGrad
�
;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_2/Relu_grad/ReluGrad4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������*;
_class1
/-loc:@gradients_1/v/dense_2/Relu_grad/ReluGrad*
T0
�
=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*A
_class7
53loc:@gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad
�
(gradients_1/v/dense_2/MatMul_grad/MatMulMatMul;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyv/dense_2/kernel/read*
transpose_a( *
T0*
transpose_b(*'
_output_shapes
:���������
�
*gradients_1/v/dense_2/MatMul_grad/MatMul_1MatMulv/dense_1/Relu;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
_output_shapes

:*
T0
�
2gradients_1/v/dense_2/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_2/MatMul_grad/MatMul+^gradients_1/v/dense_2/MatMul_grad/MatMul_1
�
:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_2/MatMul_grad/MatMul3^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:���������*;
_class1
/-loc:@gradients_1/v/dense_2/MatMul_grad/MatMul
�
<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_2/MatMul_grad/MatMul_13^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients_1/v/dense_2/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
�
(gradients_1/v/dense_1/Relu_grad/ReluGradReluGrad:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependencyv/dense_1/Relu*'
_output_shapes
:���������*
T0
�
.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_1/Relu_grad/ReluGrad*
_output_shapes
:*
data_formatNHWC*
T0
�
3gradients_1/v/dense_1/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_1/Relu_grad/ReluGrad
�
;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/Relu_grad/ReluGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense_1/Relu_grad/ReluGrad*'
_output_shapes
:���������
�
=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
(gradients_1/v/dense_1/MatMul_grad/MatMulMatMul;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyv/dense_1/kernel/read*
transpose_b(*
transpose_a( *
T0*'
_output_shapes
:��������� 
�
*gradients_1/v/dense_1/MatMul_grad/MatMul_1MatMulv/dense/Relu;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes

: *
T0
�
2gradients_1/v/dense_1/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_1/MatMul_grad/MatMul+^gradients_1/v/dense_1/MatMul_grad/MatMul_1
�
:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/MatMul_grad/MatMul3^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients_1/v/dense_1/MatMul_grad/MatMul*
T0*'
_output_shapes
:��������� 
�
<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_1/MatMul_grad/MatMul_13^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients_1/v/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

: 
�
&gradients_1/v/dense/Relu_grad/ReluGradReluGrad:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependencyv/dense/Relu*'
_output_shapes
:��������� *
T0
�
,gradients_1/v/dense/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_1/v/dense/Relu_grad/ReluGrad*
_output_shapes
: *
T0*
data_formatNHWC
�
1gradients_1/v/dense/BiasAdd_grad/tuple/group_depsNoOp-^gradients_1/v/dense/BiasAdd_grad/BiasAddGrad'^gradients_1/v/dense/Relu_grad/ReluGrad
�
9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/Relu_grad/ReluGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/v/dense/Relu_grad/ReluGrad*'
_output_shapes
:��������� *
T0
�
;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Identity,gradients_1/v/dense/BiasAdd_grad/BiasAddGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
: *?
_class5
31loc:@gradients_1/v/dense/BiasAdd_grad/BiasAddGrad*
T0
�
&gradients_1/v/dense/MatMul_grad/MatMulMatMul9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyv/dense/kernel/read*
transpose_b(*(
_output_shapes
:����������*
transpose_a( *
T0
�
(gradients_1/v/dense/MatMul_grad/MatMul_1MatMul	v/Reshape9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes
:	� *
T0
�
0gradients_1/v/dense/MatMul_grad/tuple/group_depsNoOp'^gradients_1/v/dense/MatMul_grad/MatMul)^gradients_1/v/dense/MatMul_grad/MatMul_1
�
8gradients_1/v/dense/MatMul_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/MatMul_grad/MatMul1^gradients_1/v/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:����������*
T0*9
_class/
-+loc:@gradients_1/v/dense/MatMul_grad/MatMul
�
:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Identity(gradients_1/v/dense/MatMul_grad/MatMul_11^gradients_1/v/dense/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	� *;
_class1
/-loc:@gradients_1/v/dense/MatMul_grad/MatMul_1
�
beta1_power_1/initial_valueConst*
_class
loc:@v/dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power_1
VariableV2*
_output_shapes
: *
	container *
_class
loc:@v/dense/bias*
shape: *
dtype0*
shared_name 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
validate_shape(
o
beta1_power_1/readIdentitybeta1_power_1*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0
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
loc:@v/dense/bias*
_output_shapes
: *
	container *
shape: *
dtype0*
shared_name 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(
o
beta2_power_1/readIdentitybeta2_power_1*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
�
5v/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"<      *
dtype0*!
_class
loc:@v/dense/kernel
�
+v/dense/kernel/Adam/Initializer/zeros/ConstConst*!
_class
loc:@v/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
%v/dense/kernel/Adam/Initializer/zerosFill5v/dense/kernel/Adam/Initializer/zeros/shape_as_tensor+v/dense/kernel/Adam/Initializer/zeros/Const*

index_type0*
_output_shapes
:	� *!
_class
loc:@v/dense/kernel*
T0
�
v/dense/kernel/Adam
VariableV2*!
_class
loc:@v/dense/kernel*
	container *
_output_shapes
:	� *
shared_name *
shape:	� *
dtype0
�
v/dense/kernel/Adam/AssignAssignv/dense/kernel/Adam%v/dense/kernel/Adam/Initializer/zeros*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes
:	� 
�
v/dense/kernel/Adam/readIdentityv/dense/kernel/Adam*
_output_shapes
:	� *
T0*!
_class
loc:@v/dense/kernel
�
7v/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"<      *
_output_shapes
:*
dtype0*!
_class
loc:@v/dense/kernel
�
-v/dense/kernel/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@v/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
�
'v/dense/kernel/Adam_1/Initializer/zerosFill7v/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor-v/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	� *
T0*

index_type0*!
_class
loc:@v/dense/kernel
�
v/dense/kernel/Adam_1
VariableV2*
shared_name *
	container *!
_class
loc:@v/dense/kernel*
dtype0*
shape:	� *
_output_shapes
:	� 
�
v/dense/kernel/Adam_1/AssignAssignv/dense/kernel/Adam_1'v/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes
:	� *
validate_shape(*!
_class
loc:@v/dense/kernel*
use_locking(*
T0
�
v/dense/kernel/Adam_1/readIdentityv/dense/kernel/Adam_1*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes
:	� 
�
#v/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
: *
_class
loc:@v/dense/bias*
valueB *    
�
v/dense/bias/Adam
VariableV2*
_class
loc:@v/dense/bias*
shared_name *
_output_shapes
: *
dtype0*
	container *
shape: 
�
v/dense/bias/Adam/AssignAssignv/dense/bias/Adam#v/dense/bias/Adam/Initializer/zeros*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(
{
v/dense/bias/Adam/readIdentityv/dense/bias/Adam*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
�
%v/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes
: *
_class
loc:@v/dense/bias*
valueB *    *
dtype0
�
v/dense/bias/Adam_1
VariableV2*
dtype0*
shape: *
_class
loc:@v/dense/bias*
shared_name *
_output_shapes
: *
	container 
�
v/dense/bias/Adam_1/AssignAssignv/dense/bias/Adam_1%v/dense/bias/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias*
use_locking(

v/dense/bias/Adam_1/readIdentityv/dense/bias/Adam_1*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
�
'v/dense_1/kernel/Adam/Initializer/zerosConst*
valueB *    *
_output_shapes

: *
dtype0*#
_class
loc:@v/dense_1/kernel
�
v/dense_1/kernel/Adam
VariableV2*
shape
: *
	container *
shared_name *#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
dtype0
�
v/dense_1/kernel/Adam/AssignAssignv/dense_1/kernel/Adam'v/dense_1/kernel/Adam/Initializer/zeros*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: 
�
v/dense_1/kernel/Adam/readIdentityv/dense_1/kernel/Adam*
_output_shapes

: *
T0*#
_class
loc:@v/dense_1/kernel
�
)v/dense_1/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
valueB *    
�
v/dense_1/kernel/Adam_1
VariableV2*#
_class
loc:@v/dense_1/kernel*
shape
: *
_output_shapes

: *
shared_name *
dtype0*
	container 
�
v/dense_1/kernel/Adam_1/AssignAssignv/dense_1/kernel/Adam_1)v/dense_1/kernel/Adam_1/Initializer/zeros*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0*
validate_shape(*
use_locking(
�
v/dense_1/kernel/Adam_1/readIdentityv/dense_1/kernel/Adam_1*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: 
�
%v/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *!
_class
loc:@v/dense_1/bias*
_output_shapes
:
�
v/dense_1/bias/Adam
VariableV2*
shared_name *!
_class
loc:@v/dense_1/bias*
shape:*
_output_shapes
:*
	container *
dtype0
�
v/dense_1/bias/Adam/AssignAssignv/dense_1/bias/Adam%v/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
T0
�
v/dense_1/bias/Adam/readIdentityv/dense_1/bias/Adam*
_output_shapes
:*
T0*!
_class
loc:@v/dense_1/bias
�
'v/dense_1/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*!
_class
loc:@v/dense_1/bias
�
v/dense_1/bias/Adam_1
VariableV2*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
shared_name *
	container *
shape:*
dtype0
�
v/dense_1/bias/Adam_1/AssignAssignv/dense_1/bias/Adam_1'v/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:*
validate_shape(
�
v/dense_1/bias/Adam_1/readIdentityv/dense_1/bias/Adam_1*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
T0
�
'v/dense_2/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*
valueB*    *
dtype0*#
_class
loc:@v/dense_2/kernel
�
v/dense_2/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
shared_name *
shape
:*
	container 
�
v/dense_2/kernel/Adam/AssignAssignv/dense_2/kernel/Adam'v/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0
�
v/dense_2/kernel/Adam/readIdentityv/dense_2/kernel/Adam*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:
�
)v/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:*
valueB*    *
dtype0*#
_class
loc:@v/dense_2/kernel
�
v/dense_2/kernel/Adam_1
VariableV2*
shared_name *
dtype0*
shape
:*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
	container 
�
v/dense_2/kernel/Adam_1/AssignAssignv/dense_2/kernel/Adam_1)v/dense_2/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
v/dense_2/kernel/Adam_1/readIdentityv/dense_2/kernel/Adam_1*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:
�
%v/dense_2/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*!
_class
loc:@v/dense_2/bias
�
v/dense_2/bias/Adam
VariableV2*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
shape:*
shared_name *
dtype0*
	container 
�
v/dense_2/bias/Adam/AssignAssignv/dense_2/bias/Adam%v/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0*
use_locking(
�
v/dense_2/bias/Adam/readIdentityv/dense_2/bias/Adam*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
T0
�
'v/dense_2/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *!
_class
loc:@v/dense_2/bias
�
v/dense_2/bias/Adam_1
VariableV2*!
_class
loc:@v/dense_2/bias*
shared_name *
dtype0*
shape:*
	container *
_output_shapes
:
�
v/dense_2/bias/Adam_1/AssignAssignv/dense_2/bias/Adam_1'v/dense_2/bias/Adam_1/Initializer/zeros*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
v/dense_2/bias/Adam_1/readIdentityv/dense_2/bias/Adam_1*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
'v/dense_3/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *#
_class
loc:@v/dense_3/kernel
�
v/dense_3/kernel/Adam
VariableV2*
shared_name *#
_class
loc:@v/dense_3/kernel*
shape
:*
_output_shapes

:*
dtype0*
	container 
�
v/dense_3/kernel/Adam/AssignAssignv/dense_3/kernel/Adam'v/dense_3/kernel/Adam/Initializer/zeros*
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:
�
v/dense_3/kernel/Adam/readIdentityv/dense_3/kernel/Adam*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
T0
�
)v/dense_3/kernel/Adam_1/Initializer/zerosConst*
valueB*    *#
_class
loc:@v/dense_3/kernel*
dtype0*
_output_shapes

:
�
v/dense_3/kernel/Adam_1
VariableV2*
shape
:*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
dtype0*
	container *
shared_name 
�
v/dense_3/kernel/Adam_1/AssignAssignv/dense_3/kernel/Adam_1)v/dense_3/kernel/Adam_1/Initializer/zeros*
use_locking(*#
_class
loc:@v/dense_3/kernel*
T0*
validate_shape(*
_output_shapes

:
�
v/dense_3/kernel/Adam_1/readIdentityv/dense_3/kernel/Adam_1*#
_class
loc:@v/dense_3/kernel*
T0*
_output_shapes

:
�
%v/dense_3/bias/Adam/Initializer/zerosConst*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
dtype0*
valueB*    
�
v/dense_3/bias/Adam
VariableV2*
shared_name *
_output_shapes
:*
dtype0*
shape:*!
_class
loc:@v/dense_3/bias*
	container 
�
v/dense_3/bias/Adam/AssignAssignv/dense_3/bias/Adam%v/dense_3/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_3/bias*
T0*
validate_shape(
�
v/dense_3/bias/Adam/readIdentityv/dense_3/bias/Adam*
T0*
_output_shapes
:*!
_class
loc:@v/dense_3/bias
�
'v/dense_3/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@v/dense_3/bias*
dtype0*
_output_shapes
:*
valueB*    
�
v/dense_3/bias/Adam_1
VariableV2*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
shared_name *
dtype0*
	container *
shape:
�
v/dense_3/bias/Adam_1/AssignAssignv/dense_3/bias/Adam_1'v/dense_3/bias/Adam_1/Initializer/zeros*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_3/bias
�
v/dense_3/bias/Adam_1/readIdentityv/dense_3/bias/Adam_1*
T0*
_output_shapes
:*!
_class
loc:@v/dense_3/bias
Y
Adam_1/learning_rateConst*
valueB
 *o�:*
_output_shapes
: *
dtype0
Q
Adam_1/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
Q
Adam_1/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *w�?
S
Adam_1/epsilonConst*
valueB
 *w�+2*
_output_shapes
: *
dtype0
�
&Adam_1/update_v/dense/kernel/ApplyAdam	ApplyAdamv/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
use_locking( *!
_class
loc:@v/dense/kernel*
_output_shapes
:	� 
�
$Adam_1/update_v/dense/bias/ApplyAdam	ApplyAdamv/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
use_nesterov( *
_class
loc:@v/dense/bias*
_output_shapes
: *
T0
�
(Adam_1/update_v/dense_1/kernel/ApplyAdam	ApplyAdamv/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
use_locking( *
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0
�
&Adam_1/update_v/dense_1/bias/ApplyAdam	ApplyAdamv/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1*!
_class
loc:@v/dense_1/bias*
use_locking( *
_output_shapes
:*
use_nesterov( *
T0
�
(Adam_1/update_v/dense_2/kernel/ApplyAdam	ApplyAdamv/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
T0*
use_nesterov( 
�
&Adam_1/update_v/dense_2/bias/ApplyAdam	ApplyAdamv/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
T0*
_output_shapes
:*
use_locking( *!
_class
loc:@v/dense_2/bias
�
(Adam_1/update_v/dense_3/kernel/ApplyAdam	ApplyAdamv/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon<gradients_1/v/dense_3/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:*
T0*#
_class
loc:@v/dense_3/kernel*
use_locking( 
�
&Adam_1/update_v/dense_3/bias/ApplyAdam	ApplyAdamv/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon=gradients_1/v/dense_3/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
T0*
use_locking( 
�

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam'^Adam_1/update_v/dense_3/bias/ApplyAdam)^Adam_1/update_v/dense_3/kernel/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking( 
�
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam'^Adam_1/update_v/dense_3/bias/ApplyAdam)^Adam_1/update_v/dense_3/kernel/ApplyAdam*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
T0
�
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam'^Adam_1/update_v/dense_3/bias/ApplyAdam)^Adam_1/update_v/dense_3/kernel/ApplyAdam
�
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^pi_j/dense/bias/Adam/Assign^pi_j/dense/bias/Adam_1/Assign^pi_j/dense/bias/Assign^pi_j/dense/kernel/Adam/Assign ^pi_j/dense/kernel/Adam_1/Assign^pi_j/dense/kernel/Assign^pi_j/dense_1/bias/Adam/Assign ^pi_j/dense_1/bias/Adam_1/Assign^pi_j/dense_1/bias/Assign ^pi_j/dense_1/kernel/Adam/Assign"^pi_j/dense_1/kernel/Adam_1/Assign^pi_j/dense_1/kernel/Assign^pi_j/dense_2/bias/Adam/Assign ^pi_j/dense_2/bias/Adam_1/Assign^pi_j/dense_2/bias/Assign ^pi_j/dense_2/kernel/Adam/Assign"^pi_j/dense_2/kernel/Adam_1/Assign^pi_j/dense_2/kernel/Assign^pi_j/dense_3/bias/Adam/Assign ^pi_j/dense_3/bias/Adam_1/Assign^pi_j/dense_3/bias/Assign ^pi_j/dense_3/kernel/Adam/Assign"^pi_j/dense_3/kernel/Adam_1/Assign^pi_j/dense_3/kernel/Assign^pi_n/dense/bias/Adam/Assign^pi_n/dense/bias/Adam_1/Assign^pi_n/dense/bias/Assign^pi_n/dense/kernel/Adam/Assign ^pi_n/dense/kernel/Adam_1/Assign^pi_n/dense/kernel/Assign^pi_n/dense_1/bias/Adam/Assign ^pi_n/dense_1/bias/Adam_1/Assign^pi_n/dense_1/bias/Assign ^pi_n/dense_1/kernel/Adam/Assign"^pi_n/dense_1/kernel/Adam_1/Assign^pi_n/dense_1/kernel/Assign^pi_n/dense_2/bias/Adam/Assign ^pi_n/dense_2/bias/Adam_1/Assign^pi_n/dense_2/bias/Assign ^pi_n/dense_2/kernel/Adam/Assign"^pi_n/dense_2/kernel/Adam_1/Assign^pi_n/dense_2/kernel/Assign^pi_n/dense_3/bias/Adam/Assign ^pi_n/dense_3/bias/Adam_1/Assign^pi_n/dense_3/bias/Assign ^pi_n/dense_3/kernel/Adam/Assign"^pi_n/dense_3/kernel/Adam_1/Assign^pi_n/dense_3/kernel/Assign^v/dense/bias/Adam/Assign^v/dense/bias/Adam_1/Assign^v/dense/bias/Assign^v/dense/kernel/Adam/Assign^v/dense/kernel/Adam_1/Assign^v/dense/kernel/Assign^v/dense_1/bias/Adam/Assign^v/dense_1/bias/Adam_1/Assign^v/dense_1/bias/Assign^v/dense_1/kernel/Adam/Assign^v/dense_1/kernel/Adam_1/Assign^v/dense_1/kernel/Assign^v/dense_2/bias/Adam/Assign^v/dense_2/bias/Adam_1/Assign^v/dense_2/bias/Assign^v/dense_2/kernel/Adam/Assign^v/dense_2/kernel/Adam_1/Assign^v/dense_2/kernel/Assign^v/dense_3/bias/Adam/Assign^v/dense_3/bias/Adam_1/Assign^v/dense_3/bias/Assign^v/dense_3/kernel/Adam/Assign^v/dense_3/kernel/Adam_1/Assign^v/dense_3/kernel/Assign
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_8e3cd405d6e547e7aeacb972f54e6a9e/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
save/SaveV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*
N*
_output_shapes
:*

axis 
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
�
save/RestoreV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
save/RestoreV2/shape_and_slicesConst*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*"
_class
loc:@pi_j/dense/bias*
validate_shape(*
_output_shapes
: *
T0
�
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
T0*
use_locking(*
validate_shape(*
_output_shapes
: *
_class
loc:@v/dense/bias
�
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: 
�
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save/Assign_4Assignpi_j/dense/biassave/RestoreV2:4*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
validate_shape(*
use_locking(*
T0
�
save/Assign_5Assignpi_j/dense/bias/Adamsave/RestoreV2:5*
T0*
validate_shape(*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
use_locking(
�
save/Assign_6Assignpi_j/dense/bias/Adam_1save/RestoreV2:6*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias
�
save/Assign_7Assignpi_j/dense/kernelsave/RestoreV2:7*
T0*
_output_shapes

: *
use_locking(*
validate_shape(*$
_class
loc:@pi_j/dense/kernel
�
save/Assign_8Assignpi_j/dense/kernel/Adamsave/RestoreV2:8*$
_class
loc:@pi_j/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

: *
T0
�
save/Assign_9Assignpi_j/dense/kernel/Adam_1save/RestoreV2:9*
use_locking(*
_output_shapes

: *
T0*$
_class
loc:@pi_j/dense/kernel*
validate_shape(
�
save/Assign_10Assignpi_j/dense_1/biassave/RestoreV2:10*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:
�
save/Assign_11Assignpi_j/dense_1/bias/Adamsave/RestoreV2:11*
T0*$
_class
loc:@pi_j/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes
:
�
save/Assign_12Assignpi_j/dense_1/bias/Adam_1save/RestoreV2:12*
_output_shapes
:*$
_class
loc:@pi_j/dense_1/bias*
validate_shape(*
use_locking(*
T0
�
save/Assign_13Assignpi_j/dense_1/kernelsave/RestoreV2:13*
T0*
validate_shape(*
_output_shapes

: *
use_locking(*&
_class
loc:@pi_j/dense_1/kernel
�
save/Assign_14Assignpi_j/dense_1/kernel/Adamsave/RestoreV2:14*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
use_locking(*
validate_shape(*
T0
�
save/Assign_15Assignpi_j/dense_1/kernel/Adam_1save/RestoreV2:15*
_output_shapes

: *
validate_shape(*&
_class
loc:@pi_j/dense_1/kernel*
use_locking(*
T0
�
save/Assign_16Assignpi_j/dense_2/biassave/RestoreV2:16*$
_class
loc:@pi_j/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
�
save/Assign_17Assignpi_j/dense_2/bias/Adamsave/RestoreV2:17*
_output_shapes
:*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi_j/dense_2/bias
�
save/Assign_18Assignpi_j/dense_2/bias/Adam_1save/RestoreV2:18*
use_locking(*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
�
save/Assign_19Assignpi_j/dense_2/kernelsave/RestoreV2:19*
use_locking(*
T0*
validate_shape(*
_output_shapes

:*&
_class
loc:@pi_j/dense_2/kernel
�
save/Assign_20Assignpi_j/dense_2/kernel/Adamsave/RestoreV2:20*
T0*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel*
use_locking(
�
save/Assign_21Assignpi_j/dense_2/kernel/Adam_1save/RestoreV2:21*
T0*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel*
_output_shapes

:*
use_locking(
�
save/Assign_22Assignpi_j/dense_3/biassave/RestoreV2:22*
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias*
T0*
use_locking(*
validate_shape(
�
save/Assign_23Assignpi_j/dense_3/bias/Adamsave/RestoreV2:23*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*$
_class
loc:@pi_j/dense_3/bias
�
save/Assign_24Assignpi_j/dense_3/bias/Adam_1save/RestoreV2:24*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*$
_class
loc:@pi_j/dense_3/bias
�
save/Assign_25Assignpi_j/dense_3/kernelsave/RestoreV2:25*
use_locking(*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0
�
save/Assign_26Assignpi_j/dense_3/kernel/Adamsave/RestoreV2:26*&
_class
loc:@pi_j/dense_3/kernel*
T0*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_27Assignpi_j/dense_3/kernel/Adam_1save/RestoreV2:27*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:*
T0*
validate_shape(*
use_locking(
�
save/Assign_28Assignpi_n/dense/biassave/RestoreV2:28*
validate_shape(*
_output_shapes
: *"
_class
loc:@pi_n/dense/bias*
use_locking(*
T0
�
save/Assign_29Assignpi_n/dense/bias/Adamsave/RestoreV2:29*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@pi_n/dense/bias
�
save/Assign_30Assignpi_n/dense/bias/Adam_1save/RestoreV2:30*"
_class
loc:@pi_n/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
�
save/Assign_31Assignpi_n/dense/kernelsave/RestoreV2:31*
use_locking(*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: *
validate_shape(*
T0
�
save/Assign_32Assignpi_n/dense/kernel/Adamsave/RestoreV2:32*
_output_shapes

: *
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi_n/dense/kernel
�
save/Assign_33Assignpi_n/dense/kernel/Adam_1save/RestoreV2:33*
_output_shapes

: *
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi_n/dense/kernel
�
save/Assign_34Assignpi_n/dense_1/biassave/RestoreV2:34*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_n/dense_1/bias
�
save/Assign_35Assignpi_n/dense_1/bias/Adamsave/RestoreV2:35*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_n/dense_1/bias
�
save/Assign_36Assignpi_n/dense_1/bias/Adam_1save/RestoreV2:36*
_output_shapes
:*
T0*
validate_shape(*$
_class
loc:@pi_n/dense_1/bias*
use_locking(
�
save/Assign_37Assignpi_n/dense_1/kernelsave/RestoreV2:37*
_output_shapes

: *
use_locking(*&
_class
loc:@pi_n/dense_1/kernel*
T0*
validate_shape(
�
save/Assign_38Assignpi_n/dense_1/kernel/Adamsave/RestoreV2:38*&
_class
loc:@pi_n/dense_1/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

: 
�
save/Assign_39Assignpi_n/dense_1/kernel/Adam_1save/RestoreV2:39*
_output_shapes

: *
T0*&
_class
loc:@pi_n/dense_1/kernel*
use_locking(*
validate_shape(
�
save/Assign_40Assignpi_n/dense_2/biassave/RestoreV2:40*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias
�
save/Assign_41Assignpi_n/dense_2/bias/Adamsave/RestoreV2:41*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
�
save/Assign_42Assignpi_n/dense_2/bias/Adam_1save/RestoreV2:42*
T0*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(
�
save/Assign_43Assignpi_n/dense_2/kernelsave/RestoreV2:43*
validate_shape(*
use_locking(*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:*
T0
�
save/Assign_44Assignpi_n/dense_2/kernel/Adamsave/RestoreV2:44*
use_locking(*
_output_shapes

:*
validate_shape(*
T0*&
_class
loc:@pi_n/dense_2/kernel
�
save/Assign_45Assignpi_n/dense_2/kernel/Adam_1save/RestoreV2:45*
validate_shape(*
_output_shapes

:*
T0*
use_locking(*&
_class
loc:@pi_n/dense_2/kernel
�
save/Assign_46Assignpi_n/dense_3/biassave/RestoreV2:46*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_3/bias
�
save/Assign_47Assignpi_n/dense_3/bias/Adamsave/RestoreV2:47*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
use_locking(*
T0
�
save/Assign_48Assignpi_n/dense_3/bias/Adam_1save/RestoreV2:48*
_output_shapes
:*
T0*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
use_locking(
�
save/Assign_49Assignpi_n/dense_3/kernelsave/RestoreV2:49*
use_locking(*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel*
T0*
validate_shape(
�
save/Assign_50Assignpi_n/dense_3/kernel/Adamsave/RestoreV2:50*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(*
validate_shape(
�
save/Assign_51Assignpi_n/dense_3/kernel/Adam_1save/RestoreV2:51*
use_locking(*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel*
validate_shape(*
T0
�
save/Assign_52Assignv/dense/biassave/RestoreV2:52*
_output_shapes
: *
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
�
save/Assign_53Assignv/dense/bias/Adamsave/RestoreV2:53*
use_locking(*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(
�
save/Assign_54Assignv/dense/bias/Adam_1save/RestoreV2:54*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(
�
save/Assign_55Assignv/dense/kernelsave/RestoreV2:55*!
_class
loc:@v/dense/kernel*
_output_shapes
:	� *
validate_shape(*
use_locking(*
T0
�
save/Assign_56Assignv/dense/kernel/Adamsave/RestoreV2:56*
validate_shape(*!
_class
loc:@v/dense/kernel*
_output_shapes
:	� *
T0*
use_locking(
�
save/Assign_57Assignv/dense/kernel/Adam_1save/RestoreV2:57*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	� *!
_class
loc:@v/dense/kernel
�
save/Assign_58Assignv/dense_1/biassave/RestoreV2:58*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:*
T0
�
save/Assign_59Assignv/dense_1/bias/Adamsave/RestoreV2:59*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
save/Assign_60Assignv/dense_1/bias/Adam_1save/RestoreV2:60*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
validate_shape(*
T0
�
save/Assign_61Assignv/dense_1/kernelsave/RestoreV2:61*
_output_shapes

: *
validate_shape(*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0
�
save/Assign_62Assignv/dense_1/kernel/Adamsave/RestoreV2:62*
use_locking(*#
_class
loc:@v/dense_1/kernel*
T0*
_output_shapes

: *
validate_shape(
�
save/Assign_63Assignv/dense_1/kernel/Adam_1save/RestoreV2:63*
validate_shape(*
use_locking(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0
�
save/Assign_64Assignv/dense_2/biassave/RestoreV2:64*
validate_shape(*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_65Assignv/dense_2/bias/Adamsave/RestoreV2:65*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
�
save/Assign_66Assignv/dense_2/bias/Adam_1save/RestoreV2:66*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0
�
save/Assign_67Assignv/dense_2/kernelsave/RestoreV2:67*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
use_locking(*
T0*
validate_shape(
�
save/Assign_68Assignv/dense_2/kernel/Adamsave/RestoreV2:68*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
validate_shape(*
T0*
use_locking(
�
save/Assign_69Assignv/dense_2/kernel/Adam_1save/RestoreV2:69*
T0*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(
�
save/Assign_70Assignv/dense_3/biassave/RestoreV2:70*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
use_locking(*
T0
�
save/Assign_71Assignv/dense_3/bias/Adamsave/RestoreV2:71*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias
�
save/Assign_72Assignv/dense_3/bias/Adam_1save/RestoreV2:72*
validate_shape(*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_73Assignv/dense_3/kernelsave/RestoreV2:73*#
_class
loc:@v/dense_3/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
�
save/Assign_74Assignv/dense_3/kernel/Adamsave/RestoreV2:74*
T0*#
_class
loc:@v/dense_3/kernel*
use_locking(*
_output_shapes

:*
validate_shape(
�
save/Assign_75Assignv/dense_3/kernel/Adam_1save/RestoreV2:75*
_output_shapes

:*
T0*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_3/kernel
�

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
_output_shapes
: *
dtype0
�
save_1/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5ab5f8cd4f7643538c0ec99d2845d72b/part
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_1/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
�
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
_output_shapes
:L*
dtype0
�
save_1/SaveV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename
�
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
N*
_output_shapes
:*
T0
�
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
�
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
�
save_1/RestoreV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0
�
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
T0*"
_class
loc:@pi_j/dense/bias*
use_locking(*
_output_shapes
: *
validate_shape(
�
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
validate_shape(*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
T0*
use_locking(
�
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
validate_shape(*
T0*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias
�
save_1/Assign_4Assignpi_j/dense/biassave_1/RestoreV2:4*
use_locking(*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
validate_shape(*
T0
�
save_1/Assign_5Assignpi_j/dense/bias/Adamsave_1/RestoreV2:5*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
validate_shape(*
T0*
use_locking(
�
save_1/Assign_6Assignpi_j/dense/bias/Adam_1save_1/RestoreV2:6*
T0*
use_locking(*"
_class
loc:@pi_j/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_7Assignpi_j/dense/kernelsave_1/RestoreV2:7*
use_locking(*
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel*
T0*
validate_shape(
�
save_1/Assign_8Assignpi_j/dense/kernel/Adamsave_1/RestoreV2:8*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi_j/dense/kernel*
_output_shapes

: 
�
save_1/Assign_9Assignpi_j/dense/kernel/Adam_1save_1/RestoreV2:9*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*$
_class
loc:@pi_j/dense/kernel
�
save_1/Assign_10Assignpi_j/dense_1/biassave_1/RestoreV2:10*
_output_shapes
:*
T0*$
_class
loc:@pi_j/dense_1/bias*
validate_shape(*
use_locking(
�
save_1/Assign_11Assignpi_j/dense_1/bias/Adamsave_1/RestoreV2:11*
validate_shape(*
T0*$
_class
loc:@pi_j/dense_1/bias*
use_locking(*
_output_shapes
:
�
save_1/Assign_12Assignpi_j/dense_1/bias/Adam_1save_1/RestoreV2:12*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
�
save_1/Assign_13Assignpi_j/dense_1/kernelsave_1/RestoreV2:13*
_output_shapes

: *
T0*
validate_shape(*
use_locking(*&
_class
loc:@pi_j/dense_1/kernel
�
save_1/Assign_14Assignpi_j/dense_1/kernel/Adamsave_1/RestoreV2:14*
validate_shape(*
use_locking(*
T0*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel
�
save_1/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_1/RestoreV2:15*
validate_shape(*
use_locking(*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
T0
�
save_1/Assign_16Assignpi_j/dense_2/biassave_1/RestoreV2:16*
use_locking(*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:*
validate_shape(*
T0
�
save_1/Assign_17Assignpi_j/dense_2/bias/Adamsave_1/RestoreV2:17*
validate_shape(*
T0*
_output_shapes
:*$
_class
loc:@pi_j/dense_2/bias*
use_locking(
�
save_1/Assign_18Assignpi_j/dense_2/bias/Adam_1save_1/RestoreV2:18*
use_locking(*
validate_shape(*
_output_shapes
:*
T0*$
_class
loc:@pi_j/dense_2/bias
�
save_1/Assign_19Assignpi_j/dense_2/kernelsave_1/RestoreV2:19*&
_class
loc:@pi_j/dense_2/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:
�
save_1/Assign_20Assignpi_j/dense_2/kernel/Adamsave_1/RestoreV2:20*
use_locking(*
T0*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel
�
save_1/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_1/RestoreV2:21*
validate_shape(*
_output_shapes

:*&
_class
loc:@pi_j/dense_2/kernel*
T0*
use_locking(
�
save_1/Assign_22Assignpi_j/dense_3/biassave_1/RestoreV2:22*
_output_shapes
:*
T0*$
_class
loc:@pi_j/dense_3/bias*
use_locking(*
validate_shape(
�
save_1/Assign_23Assignpi_j/dense_3/bias/Adamsave_1/RestoreV2:23*
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias*
validate_shape(*
use_locking(*
T0
�
save_1/Assign_24Assignpi_j/dense_3/bias/Adam_1save_1/RestoreV2:24*$
_class
loc:@pi_j/dense_3/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_1/Assign_25Assignpi_j/dense_3/kernelsave_1/RestoreV2:25*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel*
use_locking(*
validate_shape(*
T0
�
save_1/Assign_26Assignpi_j/dense_3/kernel/Adamsave_1/RestoreV2:26*
use_locking(*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel*
T0*
validate_shape(
�
save_1/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_1/RestoreV2:27*
use_locking(*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_j/dense_3/kernel*
T0
�
save_1/Assign_28Assignpi_n/dense/biassave_1/RestoreV2:28*
_output_shapes
: *
T0*
use_locking(*"
_class
loc:@pi_n/dense/bias*
validate_shape(
�
save_1/Assign_29Assignpi_n/dense/bias/Adamsave_1/RestoreV2:29*
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi_n/dense/bias*
_output_shapes
: 
�
save_1/Assign_30Assignpi_n/dense/bias/Adam_1save_1/RestoreV2:30*
validate_shape(*
T0*"
_class
loc:@pi_n/dense/bias*
_output_shapes
: *
use_locking(
�
save_1/Assign_31Assignpi_n/dense/kernelsave_1/RestoreV2:31*
T0*
use_locking(*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: *
validate_shape(
�
save_1/Assign_32Assignpi_n/dense/kernel/Adamsave_1/RestoreV2:32*
T0*
_output_shapes

: *
use_locking(*
validate_shape(*$
_class
loc:@pi_n/dense/kernel
�
save_1/Assign_33Assignpi_n/dense/kernel/Adam_1save_1/RestoreV2:33*$
_class
loc:@pi_n/dense/kernel*
use_locking(*
T0*
_output_shapes

: *
validate_shape(
�
save_1/Assign_34Assignpi_n/dense_1/biassave_1/RestoreV2:34*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:*
T0*
use_locking(*
validate_shape(
�
save_1/Assign_35Assignpi_n/dense_1/bias/Adamsave_1/RestoreV2:35*
T0*
use_locking(*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:*
validate_shape(
�
save_1/Assign_36Assignpi_n/dense_1/bias/Adam_1save_1/RestoreV2:36*
_output_shapes
:*$
_class
loc:@pi_n/dense_1/bias*
use_locking(*
T0*
validate_shape(
�
save_1/Assign_37Assignpi_n/dense_1/kernelsave_1/RestoreV2:37*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
T0*
use_locking(*
validate_shape(
�
save_1/Assign_38Assignpi_n/dense_1/kernel/Adamsave_1/RestoreV2:38*
validate_shape(*
_output_shapes

: *
use_locking(*&
_class
loc:@pi_n/dense_1/kernel*
T0
�
save_1/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_1/RestoreV2:39*
use_locking(*
T0*
_output_shapes

: *
validate_shape(*&
_class
loc:@pi_n/dense_1/kernel
�
save_1/Assign_40Assignpi_n/dense_2/biassave_1/RestoreV2:40*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
T0
�
save_1/Assign_41Assignpi_n/dense_2/bias/Adamsave_1/RestoreV2:41*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
T0*
use_locking(
�
save_1/Assign_42Assignpi_n/dense_2/bias/Adam_1save_1/RestoreV2:42*
T0*$
_class
loc:@pi_n/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_1/Assign_43Assignpi_n/dense_2/kernelsave_1/RestoreV2:43*
use_locking(*
validate_shape(*
_output_shapes

:*
T0*&
_class
loc:@pi_n/dense_2/kernel
�
save_1/Assign_44Assignpi_n/dense_2/kernel/Adamsave_1/RestoreV2:44*
_output_shapes

:*
use_locking(*
validate_shape(*&
_class
loc:@pi_n/dense_2/kernel*
T0
�
save_1/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_1/RestoreV2:45*
_output_shapes

:*
use_locking(*
T0*
validate_shape(*&
_class
loc:@pi_n/dense_2/kernel
�
save_1/Assign_46Assignpi_n/dense_3/biassave_1/RestoreV2:46*
T0*$
_class
loc:@pi_n/dense_3/bias*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_1/Assign_47Assignpi_n/dense_3/bias/Adamsave_1/RestoreV2:47*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
T0*
use_locking(
�
save_1/Assign_48Assignpi_n/dense_3/bias/Adam_1save_1/RestoreV2:48*$
_class
loc:@pi_n/dense_3/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(
�
save_1/Assign_49Assignpi_n/dense_3/kernelsave_1/RestoreV2:49*
use_locking(*
validate_shape(*
_output_shapes

:*
T0*&
_class
loc:@pi_n/dense_3/kernel
�
save_1/Assign_50Assignpi_n/dense_3/kernel/Adamsave_1/RestoreV2:50*
validate_shape(*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(*
T0
�
save_1/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_1/RestoreV2:51*&
_class
loc:@pi_n/dense_3/kernel*
T0*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_1/Assign_52Assignv/dense/biassave_1/RestoreV2:52*
T0*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
save_1/Assign_53Assignv/dense/bias/Adamsave_1/RestoreV2:53*
_class
loc:@v/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_54Assignv/dense/bias/Adam_1save_1/RestoreV2:54*
T0*
_output_shapes
: *
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(
�
save_1/Assign_55Assignv/dense/kernelsave_1/RestoreV2:55*
use_locking(*
_output_shapes
:	� *
validate_shape(*!
_class
loc:@v/dense/kernel*
T0
�
save_1/Assign_56Assignv/dense/kernel/Adamsave_1/RestoreV2:56*
_output_shapes
:	� *
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_1/Assign_57Assignv/dense/kernel/Adam_1save_1/RestoreV2:57*
use_locking(*
validate_shape(*
_output_shapes
:	� *
T0*!
_class
loc:@v/dense/kernel
�
save_1/Assign_58Assignv/dense_1/biassave_1/RestoreV2:58*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_1/Assign_59Assignv/dense_1/bias/Adamsave_1/RestoreV2:59*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
save_1/Assign_60Assignv/dense_1/bias/Adam_1save_1/RestoreV2:60*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
�
save_1/Assign_61Assignv/dense_1/kernelsave_1/RestoreV2:61*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(
�
save_1/Assign_62Assignv/dense_1/kernel/Adamsave_1/RestoreV2:62*
_output_shapes

: *
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(
�
save_1/Assign_63Assignv/dense_1/kernel/Adam_1save_1/RestoreV2:63*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0*
validate_shape(
�
save_1/Assign_64Assignv/dense_2/biassave_1/RestoreV2:64*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_1/Assign_65Assignv/dense_2/bias/Adamsave_1/RestoreV2:65*
use_locking(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(
�
save_1/Assign_66Assignv/dense_2/bias/Adam_1save_1/RestoreV2:66*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
�
save_1/Assign_67Assignv/dense_2/kernelsave_1/RestoreV2:67*
_output_shapes

:*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_1/Assign_68Assignv/dense_2/kernel/Adamsave_1/RestoreV2:68*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:
�
save_1/Assign_69Assignv/dense_2/kernel/Adam_1save_1/RestoreV2:69*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:*
validate_shape(
�
save_1/Assign_70Assignv/dense_3/biassave_1/RestoreV2:70*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_3/bias
�
save_1/Assign_71Assignv/dense_3/bias/Adamsave_1/RestoreV2:71*!
_class
loc:@v/dense_3/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
�
save_1/Assign_72Assignv/dense_3/bias/Adam_1save_1/RestoreV2:72*!
_class
loc:@v/dense_3/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_1/Assign_73Assignv/dense_3/kernelsave_1/RestoreV2:73*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:
�
save_1/Assign_74Assignv/dense_3/kernel/Adamsave_1/RestoreV2:74*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0*
use_locking(
�
save_1/Assign_75Assignv/dense_3/kernel/Adam_1save_1/RestoreV2:75*
_output_shapes

:*
T0*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
use_locking(
�
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_2492da0569c3478e9d9a6301443e991d/part*
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
save_2/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
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
save_2/SaveV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0
�
save_2/SaveV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
�
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*

axis *
T0*
_output_shapes
:
�
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
�
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
_output_shapes
: *
T0
�
save_2/RestoreV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
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
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
T0*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
�
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
T0*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
save_2/Assign_2Assignbeta2_powersave_2/RestoreV2:2*
T0*
validate_shape(*
_output_shapes
: *
use_locking(*"
_class
loc:@pi_j/dense/bias
�
save_2/Assign_3Assignbeta2_power_1save_2/RestoreV2:3*
T0*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
_output_shapes
: 
�
save_2/Assign_4Assignpi_j/dense/biassave_2/RestoreV2:4*
use_locking(*
T0*
validate_shape(*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias
�
save_2/Assign_5Assignpi_j/dense/bias/Adamsave_2/RestoreV2:5*"
_class
loc:@pi_j/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: *
T0
�
save_2/Assign_6Assignpi_j/dense/bias/Adam_1save_2/RestoreV2:6*"
_class
loc:@pi_j/dense/bias*
T0*
_output_shapes
: *
use_locking(*
validate_shape(
�
save_2/Assign_7Assignpi_j/dense/kernelsave_2/RestoreV2:7*$
_class
loc:@pi_j/dense/kernel*
T0*
_output_shapes

: *
validate_shape(*
use_locking(
�
save_2/Assign_8Assignpi_j/dense/kernel/Adamsave_2/RestoreV2:8*
_output_shapes

: *
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi_j/dense/kernel
�
save_2/Assign_9Assignpi_j/dense/kernel/Adam_1save_2/RestoreV2:9*
T0*
_output_shapes

: *
use_locking(*$
_class
loc:@pi_j/dense/kernel*
validate_shape(
�
save_2/Assign_10Assignpi_j/dense_1/biassave_2/RestoreV2:10*
_output_shapes
:*$
_class
loc:@pi_j/dense_1/bias*
use_locking(*
validate_shape(*
T0
�
save_2/Assign_11Assignpi_j/dense_1/bias/Adamsave_2/RestoreV2:11*
T0*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_j/dense_1/bias*
use_locking(
�
save_2/Assign_12Assignpi_j/dense_1/bias/Adam_1save_2/RestoreV2:12*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_j/dense_1/bias*
T0*
use_locking(
�
save_2/Assign_13Assignpi_j/dense_1/kernelsave_2/RestoreV2:13*
T0*
_output_shapes

: *
validate_shape(*&
_class
loc:@pi_j/dense_1/kernel*
use_locking(
�
save_2/Assign_14Assignpi_j/dense_1/kernel/Adamsave_2/RestoreV2:14*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*&
_class
loc:@pi_j/dense_1/kernel
�
save_2/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_2/RestoreV2:15*
use_locking(*&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: *
validate_shape(*
T0
�
save_2/Assign_16Assignpi_j/dense_2/biassave_2/RestoreV2:16*
T0*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_2/Assign_17Assignpi_j/dense_2/bias/Adamsave_2/RestoreV2:17*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*$
_class
loc:@pi_j/dense_2/bias
�
save_2/Assign_18Assignpi_j/dense_2/bias/Adam_1save_2/RestoreV2:18*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*$
_class
loc:@pi_j/dense_2/bias
�
save_2/Assign_19Assignpi_j/dense_2/kernelsave_2/RestoreV2:19*&
_class
loc:@pi_j/dense_2/kernel*
T0*
use_locking(*
_output_shapes

:*
validate_shape(
�
save_2/Assign_20Assignpi_j/dense_2/kernel/Adamsave_2/RestoreV2:20*
use_locking(*
T0*
_output_shapes

:*&
_class
loc:@pi_j/dense_2/kernel*
validate_shape(
�
save_2/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_2/RestoreV2:21*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel*
T0*
_output_shapes

:*
use_locking(
�
save_2/Assign_22Assignpi_j/dense_3/biassave_2/RestoreV2:22*$
_class
loc:@pi_j/dense_3/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
�
save_2/Assign_23Assignpi_j/dense_3/bias/Adamsave_2/RestoreV2:23*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi_j/dense_3/bias
�
save_2/Assign_24Assignpi_j/dense_3/bias/Adam_1save_2/RestoreV2:24*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias
�
save_2/Assign_25Assignpi_j/dense_3/kernelsave_2/RestoreV2:25*
use_locking(*
T0*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:*
validate_shape(
�
save_2/Assign_26Assignpi_j/dense_3/kernel/Adamsave_2/RestoreV2:26*
use_locking(*
validate_shape(*
T0*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:
�
save_2/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_2/RestoreV2:27*&
_class
loc:@pi_j/dense_3/kernel*
T0*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_2/Assign_28Assignpi_n/dense/biassave_2/RestoreV2:28*
validate_shape(*
_output_shapes
: *
T0*"
_class
loc:@pi_n/dense/bias*
use_locking(
�
save_2/Assign_29Assignpi_n/dense/bias/Adamsave_2/RestoreV2:29*"
_class
loc:@pi_n/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_30Assignpi_n/dense/bias/Adam_1save_2/RestoreV2:30*
_output_shapes
: *
use_locking(*"
_class
loc:@pi_n/dense/bias*
T0*
validate_shape(
�
save_2/Assign_31Assignpi_n/dense/kernelsave_2/RestoreV2:31*
validate_shape(*
T0*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel*
use_locking(
�
save_2/Assign_32Assignpi_n/dense/kernel/Adamsave_2/RestoreV2:32*
validate_shape(*
T0*
use_locking(*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel
�
save_2/Assign_33Assignpi_n/dense/kernel/Adam_1save_2/RestoreV2:33*
validate_shape(*
T0*
use_locking(*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel
�
save_2/Assign_34Assignpi_n/dense_1/biassave_2/RestoreV2:34*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_n/dense_1/bias
�
save_2/Assign_35Assignpi_n/dense_1/bias/Adamsave_2/RestoreV2:35*
validate_shape(*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:*
use_locking(*
T0
�
save_2/Assign_36Assignpi_n/dense_1/bias/Adam_1save_2/RestoreV2:36*
use_locking(*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:*
T0*
validate_shape(
�
save_2/Assign_37Assignpi_n/dense_1/kernelsave_2/RestoreV2:37*&
_class
loc:@pi_n/dense_1/kernel*
_output_shapes

: *
use_locking(*
T0*
validate_shape(
�
save_2/Assign_38Assignpi_n/dense_1/kernel/Adamsave_2/RestoreV2:38*
use_locking(*&
_class
loc:@pi_n/dense_1/kernel*
_output_shapes

: *
validate_shape(*
T0
�
save_2/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_2/RestoreV2:39*
use_locking(*&
_class
loc:@pi_n/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

: 
�
save_2/Assign_40Assignpi_n/dense_2/biassave_2/RestoreV2:40*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
�
save_2/Assign_41Assignpi_n/dense_2/bias/Adamsave_2/RestoreV2:41*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias
�
save_2/Assign_42Assignpi_n/dense_2/bias/Adam_1save_2/RestoreV2:42*
validate_shape(*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
T0*
_output_shapes
:
�
save_2/Assign_43Assignpi_n/dense_2/kernelsave_2/RestoreV2:43*&
_class
loc:@pi_n/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes

:*
T0
�
save_2/Assign_44Assignpi_n/dense_2/kernel/Adamsave_2/RestoreV2:44*
validate_shape(*
_output_shapes

:*
T0*
use_locking(*&
_class
loc:@pi_n/dense_2/kernel
�
save_2/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_2/RestoreV2:45*&
_class
loc:@pi_n/dense_2/kernel*
T0*
validate_shape(*
_output_shapes

:*
use_locking(
�
save_2/Assign_46Assignpi_n/dense_3/biassave_2/RestoreV2:46*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_n/dense_3/bias
�
save_2/Assign_47Assignpi_n/dense_3/bias/Adamsave_2/RestoreV2:47*
use_locking(*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_3/bias*
validate_shape(
�
save_2/Assign_48Assignpi_n/dense_3/bias/Adam_1save_2/RestoreV2:48*
validate_shape(*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_3/bias*
use_locking(
�
save_2/Assign_49Assignpi_n/dense_3/kernelsave_2/RestoreV2:49*
validate_shape(*
use_locking(*
T0*&
_class
loc:@pi_n/dense_3/kernel*
_output_shapes

:
�
save_2/Assign_50Assignpi_n/dense_3/kernel/Adamsave_2/RestoreV2:50*&
_class
loc:@pi_n/dense_3/kernel*
T0*
validate_shape(*
_output_shapes

:*
use_locking(
�
save_2/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_2/RestoreV2:51*
_output_shapes

:*
validate_shape(*
T0*
use_locking(*&
_class
loc:@pi_n/dense_3/kernel
�
save_2/Assign_52Assignv/dense/biassave_2/RestoreV2:52*
T0*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
save_2/Assign_53Assignv/dense/bias/Adamsave_2/RestoreV2:53*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias*
T0
�
save_2/Assign_54Assignv/dense/bias/Adam_1save_2/RestoreV2:54*
_class
loc:@v/dense/bias*
T0*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_2/Assign_55Assignv/dense/kernelsave_2/RestoreV2:55*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
_output_shapes
:	� 
�
save_2/Assign_56Assignv/dense/kernel/Adamsave_2/RestoreV2:56*
T0*
_output_shapes
:	� *
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_2/Assign_57Assignv/dense/kernel/Adam_1save_2/RestoreV2:57*
_output_shapes
:	� *
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
validate_shape(
�
save_2/Assign_58Assignv/dense_1/biassave_2/RestoreV2:58*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
use_locking(*
T0
�
save_2/Assign_59Assignv/dense_1/bias/Adamsave_2/RestoreV2:59*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:
�
save_2/Assign_60Assignv/dense_1/bias/Adam_1save_2/RestoreV2:60*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias
�
save_2/Assign_61Assignv/dense_1/kernelsave_2/RestoreV2:61*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(
�
save_2/Assign_62Assignv/dense_1/kernel/Adamsave_2/RestoreV2:62*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

: 
�
save_2/Assign_63Assignv/dense_1/kernel/Adam_1save_2/RestoreV2:63*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_2/Assign_64Assignv/dense_2/biassave_2/RestoreV2:64*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_2/bias
�
save_2/Assign_65Assignv/dense_2/bias/Adamsave_2/RestoreV2:65*!
_class
loc:@v/dense_2/bias*
use_locking(*
_output_shapes
:*
validate_shape(*
T0
�
save_2/Assign_66Assignv/dense_2/bias/Adam_1save_2/RestoreV2:66*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias
�
save_2/Assign_67Assignv/dense_2/kernelsave_2/RestoreV2:67*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:*
T0*
validate_shape(
�
save_2/Assign_68Assignv/dense_2/kernel/Adamsave_2/RestoreV2:68*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:*
T0
�
save_2/Assign_69Assignv/dense_2/kernel/Adam_1save_2/RestoreV2:69*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0*
use_locking(
�
save_2/Assign_70Assignv/dense_3/biassave_2/RestoreV2:70*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
�
save_2/Assign_71Assignv/dense_3/bias/Adamsave_2/RestoreV2:71*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_3/bias
�
save_2/Assign_72Assignv/dense_3/bias/Adam_1save_2/RestoreV2:72*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_3/bias*
T0*
_output_shapes
:
�
save_2/Assign_73Assignv/dense_3/kernelsave_2/RestoreV2:73*#
_class
loc:@v/dense_3/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:
�
save_2/Assign_74Assignv/dense_3/kernel/Adamsave_2/RestoreV2:74*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@v/dense_3/kernel*
validate_shape(
�
save_2/Assign_75Assignv/dense_3/kernel/Adam_1save_2/RestoreV2:75*
_output_shapes

:*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
T0
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
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
dtype0*
shape: *
_output_shapes
: 
�
save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_600013e00d274900a552474bf1118d67/part*
_output_shapes
: *
dtype0
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_3/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_3/ShardedFilename/shardConst*
value	B : *
_output_shapes
: *
dtype0
�
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
�
save_3/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
save_3/SaveV2/shape_and_slicesConst*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L
�
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
�
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
_output_shapes
:*

axis *
T0*
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
save_3/RestoreV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1
�
!save_3/RestoreV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L
�
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
T0*
validate_shape(*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
use_locking(
�
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
: 
�
save_3/Assign_2Assignbeta2_powersave_3/RestoreV2:2*"
_class
loc:@pi_j/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
�
save_3/Assign_3Assignbeta2_power_1save_3/RestoreV2:3*
_output_shapes
: *
use_locking(*
validate_shape(*
T0*
_class
loc:@v/dense/bias
�
save_3/Assign_4Assignpi_j/dense/biassave_3/RestoreV2:4*
use_locking(*
_output_shapes
: *
validate_shape(*
T0*"
_class
loc:@pi_j/dense/bias
�
save_3/Assign_5Assignpi_j/dense/bias/Adamsave_3/RestoreV2:5*
use_locking(*
validate_shape(*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
T0
�
save_3/Assign_6Assignpi_j/dense/bias/Adam_1save_3/RestoreV2:6*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@pi_j/dense/bias
�
save_3/Assign_7Assignpi_j/dense/kernelsave_3/RestoreV2:7*$
_class
loc:@pi_j/dense/kernel*
use_locking(*
_output_shapes

: *
T0*
validate_shape(
�
save_3/Assign_8Assignpi_j/dense/kernel/Adamsave_3/RestoreV2:8*
_output_shapes

: *
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi_j/dense/kernel
�
save_3/Assign_9Assignpi_j/dense/kernel/Adam_1save_3/RestoreV2:9*
use_locking(*
T0*
validate_shape(*
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel
�
save_3/Assign_10Assignpi_j/dense_1/biassave_3/RestoreV2:10*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
save_3/Assign_11Assignpi_j/dense_1/bias/Adamsave_3/RestoreV2:11*
T0*
use_locking(*$
_class
loc:@pi_j/dense_1/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_12Assignpi_j/dense_1/bias/Adam_1save_3/RestoreV2:12*
T0*
_output_shapes
:*$
_class
loc:@pi_j/dense_1/bias*
use_locking(*
validate_shape(
�
save_3/Assign_13Assignpi_j/dense_1/kernelsave_3/RestoreV2:13*
_output_shapes

: *
validate_shape(*&
_class
loc:@pi_j/dense_1/kernel*
T0*
use_locking(
�
save_3/Assign_14Assignpi_j/dense_1/kernel/Adamsave_3/RestoreV2:14*
_output_shapes

: *
T0*&
_class
loc:@pi_j/dense_1/kernel*
validate_shape(*
use_locking(
�
save_3/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_3/RestoreV2:15*
T0*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
validate_shape(*
use_locking(
�
save_3/Assign_16Assignpi_j/dense_2/biassave_3/RestoreV2:16*
T0*$
_class
loc:@pi_j/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_3/Assign_17Assignpi_j/dense_2/bias/Adamsave_3/RestoreV2:17*$
_class
loc:@pi_j/dense_2/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes
:
�
save_3/Assign_18Assignpi_j/dense_2/bias/Adam_1save_3/RestoreV2:18*
T0*
_output_shapes
:*$
_class
loc:@pi_j/dense_2/bias*
validate_shape(*
use_locking(
�
save_3/Assign_19Assignpi_j/dense_2/kernelsave_3/RestoreV2:19*
_output_shapes

:*
validate_shape(*
T0*&
_class
loc:@pi_j/dense_2/kernel*
use_locking(
�
save_3/Assign_20Assignpi_j/dense_2/kernel/Adamsave_3/RestoreV2:20*
use_locking(*
T0*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel
�
save_3/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_3/RestoreV2:21*&
_class
loc:@pi_j/dense_2/kernel*
_output_shapes

:*
use_locking(*
validate_shape(*
T0
�
save_3/Assign_22Assignpi_j/dense_3/biassave_3/RestoreV2:22*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_j/dense_3/bias
�
save_3/Assign_23Assignpi_j/dense_3/bias/Adamsave_3/RestoreV2:23*
T0*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_j/dense_3/bias*
use_locking(
�
save_3/Assign_24Assignpi_j/dense_3/bias/Adam_1save_3/RestoreV2:24*
T0*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias*
use_locking(
�
save_3/Assign_25Assignpi_j/dense_3/kernelsave_3/RestoreV2:25*
use_locking(*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0
�
save_3/Assign_26Assignpi_j/dense_3/kernel/Adamsave_3/RestoreV2:26*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel*
use_locking(*
T0*
validate_shape(
�
save_3/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_3/RestoreV2:27*&
_class
loc:@pi_j/dense_3/kernel*
T0*
validate_shape(*
_output_shapes

:*
use_locking(
�
save_3/Assign_28Assignpi_n/dense/biassave_3/RestoreV2:28*
T0*
validate_shape(*"
_class
loc:@pi_n/dense/bias*
_output_shapes
: *
use_locking(
�
save_3/Assign_29Assignpi_n/dense/bias/Adamsave_3/RestoreV2:29*
use_locking(*
T0*"
_class
loc:@pi_n/dense/bias*
_output_shapes
: *
validate_shape(
�
save_3/Assign_30Assignpi_n/dense/bias/Adam_1save_3/RestoreV2:30*"
_class
loc:@pi_n/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
�
save_3/Assign_31Assignpi_n/dense/kernelsave_3/RestoreV2:31*
T0*$
_class
loc:@pi_n/dense/kernel*
use_locking(*
_output_shapes

: *
validate_shape(
�
save_3/Assign_32Assignpi_n/dense/kernel/Adamsave_3/RestoreV2:32*
use_locking(*
T0*
_output_shapes

: *
validate_shape(*$
_class
loc:@pi_n/dense/kernel
�
save_3/Assign_33Assignpi_n/dense/kernel/Adam_1save_3/RestoreV2:33*$
_class
loc:@pi_n/dense/kernel*
validate_shape(*
T0*
_output_shapes

: *
use_locking(
�
save_3/Assign_34Assignpi_n/dense_1/biassave_3/RestoreV2:34*$
_class
loc:@pi_n/dense_1/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_3/Assign_35Assignpi_n/dense_1/bias/Adamsave_3/RestoreV2:35*
validate_shape(*$
_class
loc:@pi_n/dense_1/bias*
T0*
use_locking(*
_output_shapes
:
�
save_3/Assign_36Assignpi_n/dense_1/bias/Adam_1save_3/RestoreV2:36*
_output_shapes
:*
T0*
validate_shape(*$
_class
loc:@pi_n/dense_1/bias*
use_locking(
�
save_3/Assign_37Assignpi_n/dense_1/kernelsave_3/RestoreV2:37*
use_locking(*&
_class
loc:@pi_n/dense_1/kernel*
validate_shape(*
_output_shapes

: *
T0
�
save_3/Assign_38Assignpi_n/dense_1/kernel/Adamsave_3/RestoreV2:38*
use_locking(*
validate_shape(*&
_class
loc:@pi_n/dense_1/kernel*
_output_shapes

: *
T0
�
save_3/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_3/RestoreV2:39*
T0*
validate_shape(*&
_class
loc:@pi_n/dense_1/kernel*
_output_shapes

: *
use_locking(
�
save_3/Assign_40Assignpi_n/dense_2/biassave_3/RestoreV2:40*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias
�
save_3/Assign_41Assignpi_n/dense_2/bias/Adamsave_3/RestoreV2:41*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_3/Assign_42Assignpi_n/dense_2/bias/Adam_1save_3/RestoreV2:42*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi_n/dense_2/bias
�
save_3/Assign_43Assignpi_n/dense_2/kernelsave_3/RestoreV2:43*
_output_shapes

:*
T0*
use_locking(*
validate_shape(*&
_class
loc:@pi_n/dense_2/kernel
�
save_3/Assign_44Assignpi_n/dense_2/kernel/Adamsave_3/RestoreV2:44*
_output_shapes

:*&
_class
loc:@pi_n/dense_2/kernel*
use_locking(*
T0*
validate_shape(
�
save_3/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_3/RestoreV2:45*
validate_shape(*
_output_shapes

:*&
_class
loc:@pi_n/dense_2/kernel*
T0*
use_locking(
�
save_3/Assign_46Assignpi_n/dense_3/biassave_3/RestoreV2:46*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
use_locking(*
T0*
_output_shapes
:
�
save_3/Assign_47Assignpi_n/dense_3/bias/Adamsave_3/RestoreV2:47*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi_n/dense_3/bias
�
save_3/Assign_48Assignpi_n/dense_3/bias/Adam_1save_3/RestoreV2:48*
T0*
use_locking(*$
_class
loc:@pi_n/dense_3/bias*
_output_shapes
:*
validate_shape(
�
save_3/Assign_49Assignpi_n/dense_3/kernelsave_3/RestoreV2:49*
use_locking(*
validate_shape(*
_output_shapes

:*
T0*&
_class
loc:@pi_n/dense_3/kernel
�
save_3/Assign_50Assignpi_n/dense_3/kernel/Adamsave_3/RestoreV2:50*
_output_shapes

:*
use_locking(*
validate_shape(*
T0*&
_class
loc:@pi_n/dense_3/kernel
�
save_3/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_3/RestoreV2:51*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(*
T0
�
save_3/Assign_52Assignv/dense/biassave_3/RestoreV2:52*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
�
save_3/Assign_53Assignv/dense/bias/Adamsave_3/RestoreV2:53*
validate_shape(*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias*
T0
�
save_3/Assign_54Assignv/dense/bias/Adam_1save_3/RestoreV2:54*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
_output_shapes
: 
�
save_3/Assign_55Assignv/dense/kernelsave_3/RestoreV2:55*
_output_shapes
:	� *!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
validate_shape(
�
save_3/Assign_56Assignv/dense/kernel/Adamsave_3/RestoreV2:56*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	� *
T0
�
save_3/Assign_57Assignv/dense/kernel/Adam_1save_3/RestoreV2:57*
use_locking(*
_output_shapes
:	� *
T0*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_3/Assign_58Assignv/dense_1/biassave_3/RestoreV2:58*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
save_3/Assign_59Assignv/dense_1/bias/Adamsave_3/RestoreV2:59*!
_class
loc:@v/dense_1/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_3/Assign_60Assignv/dense_1/bias/Adam_1save_3/RestoreV2:60*
T0*
validate_shape(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
use_locking(
�
save_3/Assign_61Assignv/dense_1/kernelsave_3/RestoreV2:61*#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

: 
�
save_3/Assign_62Assignv/dense_1/kernel/Adamsave_3/RestoreV2:62*#
_class
loc:@v/dense_1/kernel*
T0*
validate_shape(*
_output_shapes

: *
use_locking(
�
save_3/Assign_63Assignv/dense_1/kernel/Adam_1save_3/RestoreV2:63*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
use_locking(
�
save_3/Assign_64Assignv/dense_2/biassave_3/RestoreV2:64*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0
�
save_3/Assign_65Assignv/dense_2/bias/Adamsave_3/RestoreV2:65*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
�
save_3/Assign_66Assignv/dense_2/bias/Adam_1save_3/RestoreV2:66*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_67Assignv/dense_2/kernelsave_3/RestoreV2:67*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
validate_shape(
�
save_3/Assign_68Assignv/dense_2/kernel/Adamsave_3/RestoreV2:68*
T0*
_output_shapes

:*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_3/Assign_69Assignv/dense_2/kernel/Adam_1save_3/RestoreV2:69*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_3/Assign_70Assignv/dense_3/biassave_3/RestoreV2:70*
_output_shapes
:*
T0*!
_class
loc:@v/dense_3/bias*
validate_shape(*
use_locking(
�
save_3/Assign_71Assignv/dense_3/bias/Adamsave_3/RestoreV2:71*
use_locking(*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_3/bias
�
save_3/Assign_72Assignv/dense_3/bias/Adam_1save_3/RestoreV2:72*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_3/bias*
use_locking(*
T0
�
save_3/Assign_73Assignv/dense_3/kernelsave_3/RestoreV2:73*
T0*
use_locking(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
validate_shape(
�
save_3/Assign_74Assignv/dense_3/kernel/Adamsave_3/RestoreV2:74*
use_locking(*
T0*
validate_shape(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:
�
save_3/Assign_75Assignv/dense_3/kernel/Adam_1save_3/RestoreV2:75*
use_locking(*
validate_shape(*
_output_shapes

:*
T0*#
_class
loc:@v/dense_3/kernel
�
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_51^save_3/Assign_52^save_3/Assign_53^save_3/Assign_54^save_3/Assign_55^save_3/Assign_56^save_3/Assign_57^save_3/Assign_58^save_3/Assign_59^save_3/Assign_6^save_3/Assign_60^save_3/Assign_61^save_3/Assign_62^save_3/Assign_63^save_3/Assign_64^save_3/Assign_65^save_3/Assign_66^save_3/Assign_67^save_3/Assign_68^save_3/Assign_69^save_3/Assign_7^save_3/Assign_70^save_3/Assign_71^save_3/Assign_72^save_3/Assign_73^save_3/Assign_74^save_3/Assign_75^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_4/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_be8e558e18964887a534da3ddce212c4/part
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_4/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_4/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
�
save_4/SaveV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0
�
save_4/SaveV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
_output_shapes
: *)
_class
loc:@save_4/ShardedFilename*
T0
�
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
_output_shapes
:*
N*

axis *
T0
�
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
�
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
�
save_4/RestoreV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1
�
!save_4/RestoreV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
T0*
use_locking(*
_output_shapes
: *
validate_shape(*"
_class
loc:@pi_j/dense/bias
�
save_4/Assign_1Assignbeta1_power_1save_4/RestoreV2:1*
validate_shape(*
_class
loc:@v/dense/bias*
use_locking(*
T0*
_output_shapes
: 
�
save_4/Assign_2Assignbeta2_powersave_4/RestoreV2:2*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*"
_class
loc:@pi_j/dense/bias
�
save_4/Assign_3Assignbeta2_power_1save_4/RestoreV2:3*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
save_4/Assign_4Assignpi_j/dense/biassave_4/RestoreV2:4*
validate_shape(*
T0*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
use_locking(
�
save_4/Assign_5Assignpi_j/dense/bias/Adamsave_4/RestoreV2:5*
validate_shape(*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
T0*
use_locking(
�
save_4/Assign_6Assignpi_j/dense/bias/Adam_1save_4/RestoreV2:6*
T0*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
�
save_4/Assign_7Assignpi_j/dense/kernelsave_4/RestoreV2:7*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*$
_class
loc:@pi_j/dense/kernel
�
save_4/Assign_8Assignpi_j/dense/kernel/Adamsave_4/RestoreV2:8*
use_locking(*
T0*
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel*
validate_shape(
�
save_4/Assign_9Assignpi_j/dense/kernel/Adam_1save_4/RestoreV2:9*
use_locking(*
validate_shape(*
_output_shapes

: *
T0*$
_class
loc:@pi_j/dense/kernel
�
save_4/Assign_10Assignpi_j/dense_1/biassave_4/RestoreV2:10*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi_j/dense_1/bias
�
save_4/Assign_11Assignpi_j/dense_1/bias/Adamsave_4/RestoreV2:11*
T0*$
_class
loc:@pi_j/dense_1/bias*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_4/Assign_12Assignpi_j/dense_1/bias/Adam_1save_4/RestoreV2:12*
T0*$
_class
loc:@pi_j/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_4/Assign_13Assignpi_j/dense_1/kernelsave_4/RestoreV2:13*
use_locking(*&
_class
loc:@pi_j/dense_1/kernel*
validate_shape(*
_output_shapes

: *
T0
�
save_4/Assign_14Assignpi_j/dense_1/kernel/Adamsave_4/RestoreV2:14*
validate_shape(*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
use_locking(*
T0
�
save_4/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_4/RestoreV2:15*&
_class
loc:@pi_j/dense_1/kernel*
use_locking(*
_output_shapes

: *
validate_shape(*
T0
�
save_4/Assign_16Assignpi_j/dense_2/biassave_4/RestoreV2:16*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:
�
save_4/Assign_17Assignpi_j/dense_2/bias/Adamsave_4/RestoreV2:17*
T0*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_4/Assign_18Assignpi_j/dense_2/bias/Adam_1save_4/RestoreV2:18*$
_class
loc:@pi_j/dense_2/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
:
�
save_4/Assign_19Assignpi_j/dense_2/kernelsave_4/RestoreV2:19*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*&
_class
loc:@pi_j/dense_2/kernel
�
save_4/Assign_20Assignpi_j/dense_2/kernel/Adamsave_4/RestoreV2:20*&
_class
loc:@pi_j/dense_2/kernel*
T0*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_4/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_4/RestoreV2:21*&
_class
loc:@pi_j/dense_2/kernel*
use_locking(*
_output_shapes

:*
T0*
validate_shape(
�
save_4/Assign_22Assignpi_j/dense_3/biassave_4/RestoreV2:22*
validate_shape(*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_j/dense_3/bias*
T0
�
save_4/Assign_23Assignpi_j/dense_3/bias/Adamsave_4/RestoreV2:23*
validate_shape(*
T0*$
_class
loc:@pi_j/dense_3/bias*
_output_shapes
:*
use_locking(
�
save_4/Assign_24Assignpi_j/dense_3/bias/Adam_1save_4/RestoreV2:24*$
_class
loc:@pi_j/dense_3/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_4/Assign_25Assignpi_j/dense_3/kernelsave_4/RestoreV2:25*
T0*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_4/Assign_26Assignpi_j/dense_3/kernel/Adamsave_4/RestoreV2:26*
T0*&
_class
loc:@pi_j/dense_3/kernel*
use_locking(*
validate_shape(*
_output_shapes

:
�
save_4/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_4/RestoreV2:27*
T0*&
_class
loc:@pi_j/dense_3/kernel*
use_locking(*
_output_shapes

:*
validate_shape(
�
save_4/Assign_28Assignpi_n/dense/biassave_4/RestoreV2:28*
_output_shapes
: *"
_class
loc:@pi_n/dense/bias*
T0*
validate_shape(*
use_locking(
�
save_4/Assign_29Assignpi_n/dense/bias/Adamsave_4/RestoreV2:29*
T0*
_output_shapes
: *
use_locking(*"
_class
loc:@pi_n/dense/bias*
validate_shape(
�
save_4/Assign_30Assignpi_n/dense/bias/Adam_1save_4/RestoreV2:30*
T0*
validate_shape(*"
_class
loc:@pi_n/dense/bias*
_output_shapes
: *
use_locking(
�
save_4/Assign_31Assignpi_n/dense/kernelsave_4/RestoreV2:31*
T0*
use_locking(*
_output_shapes

: *
validate_shape(*$
_class
loc:@pi_n/dense/kernel
�
save_4/Assign_32Assignpi_n/dense/kernel/Adamsave_4/RestoreV2:32*
use_locking(*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: *
validate_shape(*
T0
�
save_4/Assign_33Assignpi_n/dense/kernel/Adam_1save_4/RestoreV2:33*
validate_shape(*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel*
T0*
use_locking(
�
save_4/Assign_34Assignpi_n/dense_1/biassave_4/RestoreV2:34*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:
�
save_4/Assign_35Assignpi_n/dense_1/bias/Adamsave_4/RestoreV2:35*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_n/dense_1/bias
�
save_4/Assign_36Assignpi_n/dense_1/bias/Adam_1save_4/RestoreV2:36*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_1/bias*
validate_shape(*
use_locking(
�
save_4/Assign_37Assignpi_n/dense_1/kernelsave_4/RestoreV2:37*
use_locking(*&
_class
loc:@pi_n/dense_1/kernel*
_output_shapes

: *
validate_shape(*
T0
�
save_4/Assign_38Assignpi_n/dense_1/kernel/Adamsave_4/RestoreV2:38*
_output_shapes

: *
T0*
use_locking(*&
_class
loc:@pi_n/dense_1/kernel*
validate_shape(
�
save_4/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_4/RestoreV2:39*
T0*
use_locking(*
validate_shape(*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel
�
save_4/Assign_40Assignpi_n/dense_2/biassave_4/RestoreV2:40*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
T0*
use_locking(
�
save_4/Assign_41Assignpi_n/dense_2/bias/Adamsave_4/RestoreV2:41*
validate_shape(*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
T0
�
save_4/Assign_42Assignpi_n/dense_2/bias/Adam_1save_4/RestoreV2:42*$
_class
loc:@pi_n/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_4/Assign_43Assignpi_n/dense_2/kernelsave_4/RestoreV2:43*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_n/dense_2/kernel*
use_locking(*
T0
�
save_4/Assign_44Assignpi_n/dense_2/kernel/Adamsave_4/RestoreV2:44*
use_locking(*
T0*
validate_shape(*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:
�
save_4/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_4/RestoreV2:45*
use_locking(*
_output_shapes

:*&
_class
loc:@pi_n/dense_2/kernel*
T0*
validate_shape(
�
save_4/Assign_46Assignpi_n/dense_3/biassave_4/RestoreV2:46*
T0*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
_output_shapes
:*
use_locking(
�
save_4/Assign_47Assignpi_n/dense_3/bias/Adamsave_4/RestoreV2:47*
validate_shape(*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_3/bias*
use_locking(
�
save_4/Assign_48Assignpi_n/dense_3/bias/Adam_1save_4/RestoreV2:48*
_output_shapes
:*$
_class
loc:@pi_n/dense_3/bias*
use_locking(*
validate_shape(*
T0
�
save_4/Assign_49Assignpi_n/dense_3/kernelsave_4/RestoreV2:49*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel
�
save_4/Assign_50Assignpi_n/dense_3/kernel/Adamsave_4/RestoreV2:50*
T0*
validate_shape(*&
_class
loc:@pi_n/dense_3/kernel*
_output_shapes

:*
use_locking(
�
save_4/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_4/RestoreV2:51*
_output_shapes

:*
validate_shape(*
T0*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(
�
save_4/Assign_52Assignv/dense/biassave_4/RestoreV2:52*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(
�
save_4/Assign_53Assignv/dense/bias/Adamsave_4/RestoreV2:53*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(
�
save_4/Assign_54Assignv/dense/bias/Adam_1save_4/RestoreV2:54*
_output_shapes
: *
T0*
use_locking(*
validate_shape(*
_class
loc:@v/dense/bias
�
save_4/Assign_55Assignv/dense/kernelsave_4/RestoreV2:55*!
_class
loc:@v/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes
:	� *
T0
�
save_4/Assign_56Assignv/dense/kernel/Adamsave_4/RestoreV2:56*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	� 
�
save_4/Assign_57Assignv/dense/kernel/Adam_1save_4/RestoreV2:57*
_output_shapes
:	� *!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
validate_shape(
�
save_4/Assign_58Assignv/dense_1/biassave_4/RestoreV2:58*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(*
T0*
_output_shapes
:
�
save_4/Assign_59Assignv/dense_1/bias/Adamsave_4/RestoreV2:59*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes
:
�
save_4/Assign_60Assignv/dense_1/bias/Adam_1save_4/RestoreV2:60*
use_locking(*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
validate_shape(*
T0
�
save_4/Assign_61Assignv/dense_1/kernelsave_4/RestoreV2:61*
T0*
validate_shape(*
_output_shapes

: *
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_4/Assign_62Assignv/dense_1/kernel/Adamsave_4/RestoreV2:62*
_output_shapes

: *
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_4/Assign_63Assignv/dense_1/kernel/Adam_1save_4/RestoreV2:63*#
_class
loc:@v/dense_1/kernel*
use_locking(*
_output_shapes

: *
T0*
validate_shape(
�
save_4/Assign_64Assignv/dense_2/biassave_4/RestoreV2:64*
T0*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_4/Assign_65Assignv/dense_2/bias/Adamsave_4/RestoreV2:65*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
�
save_4/Assign_66Assignv/dense_2/bias/Adam_1save_4/RestoreV2:66*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_4/Assign_67Assignv/dense_2/kernelsave_4/RestoreV2:67*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:
�
save_4/Assign_68Assignv/dense_2/kernel/Adamsave_4/RestoreV2:68*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:
�
save_4/Assign_69Assignv/dense_2/kernel/Adam_1save_4/RestoreV2:69*
T0*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_4/Assign_70Assignv/dense_3/biassave_4/RestoreV2:70*
validate_shape(*!
_class
loc:@v/dense_3/bias*
T0*
use_locking(*
_output_shapes
:
�
save_4/Assign_71Assignv/dense_3/bias/Adamsave_4/RestoreV2:71*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_3/bias
�
save_4/Assign_72Assignv/dense_3/bias/Adam_1save_4/RestoreV2:72*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
use_locking(
�
save_4/Assign_73Assignv/dense_3/kernelsave_4/RestoreV2:73*
T0*
use_locking(*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel
�
save_4/Assign_74Assignv/dense_3/kernel/Adamsave_4/RestoreV2:74*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:
�
save_4/Assign_75Assignv/dense_3/kernel/Adam_1save_4/RestoreV2:75*
use_locking(*#
_class
loc:@v/dense_3/kernel*
T0*
validate_shape(*
_output_shapes

:
�
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_40^save_4/Assign_41^save_4/Assign_42^save_4/Assign_43^save_4/Assign_44^save_4/Assign_45^save_4/Assign_46^save_4/Assign_47^save_4/Assign_48^save_4/Assign_49^save_4/Assign_5^save_4/Assign_50^save_4/Assign_51^save_4/Assign_52^save_4/Assign_53^save_4/Assign_54^save_4/Assign_55^save_4/Assign_56^save_4/Assign_57^save_4/Assign_58^save_4/Assign_59^save_4/Assign_6^save_4/Assign_60^save_4/Assign_61^save_4/Assign_62^save_4/Assign_63^save_4/Assign_64^save_4/Assign_65^save_4/Assign_66^save_4/Assign_67^save_4/Assign_68^save_4/Assign_69^save_4/Assign_7^save_4/Assign_70^save_4/Assign_71^save_4/Assign_72^save_4/Assign_73^save_4/Assign_74^save_4/Assign_75^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_5/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_622e5b0104bc45008f1e16b31cc5564f/part
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_5/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
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
save_5/SaveV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
save_5/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_5/ShardedFilename
�
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*

axis *
T0*
N*
_output_shapes
:
�
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
�
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
�
save_5/RestoreV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0
�
!save_5/RestoreV2/shape_and_slicesConst*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L
�
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
T0*
use_locking(*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
validate_shape(
�
save_5/Assign_1Assignbeta1_power_1save_5/RestoreV2:1*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*
_class
loc:@v/dense/bias
�
save_5/Assign_2Assignbeta2_powersave_5/RestoreV2:2*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
use_locking(*
validate_shape(*
T0
�
save_5/Assign_3Assignbeta2_power_1save_5/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(
�
save_5/Assign_4Assignpi_j/dense/biassave_5/RestoreV2:4*"
_class
loc:@pi_j/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
�
save_5/Assign_5Assignpi_j/dense/bias/Adamsave_5/RestoreV2:5*"
_class
loc:@pi_j/dense/bias*
use_locking(*
T0*
validate_shape(*
_output_shapes
: 
�
save_5/Assign_6Assignpi_j/dense/bias/Adam_1save_5/RestoreV2:6*"
_class
loc:@pi_j/dense/bias*
use_locking(*
T0*
_output_shapes
: *
validate_shape(
�
save_5/Assign_7Assignpi_j/dense/kernelsave_5/RestoreV2:7*
validate_shape(*$
_class
loc:@pi_j/dense/kernel*
use_locking(*
T0*
_output_shapes

: 
�
save_5/Assign_8Assignpi_j/dense/kernel/Adamsave_5/RestoreV2:8*
validate_shape(*
use_locking(*$
_class
loc:@pi_j/dense/kernel*
T0*
_output_shapes

: 
�
save_5/Assign_9Assignpi_j/dense/kernel/Adam_1save_5/RestoreV2:9*
_output_shapes

: *
use_locking(*
T0*$
_class
loc:@pi_j/dense/kernel*
validate_shape(
�
save_5/Assign_10Assignpi_j/dense_1/biassave_5/RestoreV2:10*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi_j/dense_1/bias
�
save_5/Assign_11Assignpi_j/dense_1/bias/Adamsave_5/RestoreV2:11*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_j/dense_1/bias*
use_locking(*
T0
�
save_5/Assign_12Assignpi_j/dense_1/bias/Adam_1save_5/RestoreV2:12*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_j/dense_1/bias
�
save_5/Assign_13Assignpi_j/dense_1/kernelsave_5/RestoreV2:13*
use_locking(*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
T0*
validate_shape(
�
save_5/Assign_14Assignpi_j/dense_1/kernel/Adamsave_5/RestoreV2:14*
T0*&
_class
loc:@pi_j/dense_1/kernel*
validate_shape(*
_output_shapes

: *
use_locking(
�
save_5/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_5/RestoreV2:15*
use_locking(*
T0*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
validate_shape(
�
save_5/Assign_16Assignpi_j/dense_2/biassave_5/RestoreV2:16*
use_locking(*
T0*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:*
validate_shape(
�
save_5/Assign_17Assignpi_j/dense_2/bias/Adamsave_5/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@pi_j/dense_2/bias
�
save_5/Assign_18Assignpi_j/dense_2/bias/Adam_1save_5/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@pi_j/dense_2/bias
�
save_5/Assign_19Assignpi_j/dense_2/kernelsave_5/RestoreV2:19*
T0*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel*
use_locking(
�
save_5/Assign_20Assignpi_j/dense_2/kernel/Adamsave_5/RestoreV2:20*
T0*&
_class
loc:@pi_j/dense_2/kernel*
_output_shapes

:*
use_locking(*
validate_shape(
�
save_5/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_5/RestoreV2:21*
use_locking(*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel*
T0*
_output_shapes

:
�
save_5/Assign_22Assignpi_j/dense_3/biassave_5/RestoreV2:22*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi_j/dense_3/bias*
_output_shapes
:
�
save_5/Assign_23Assignpi_j/dense_3/bias/Adamsave_5/RestoreV2:23*
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias*
T0*
use_locking(*
validate_shape(
�
save_5/Assign_24Assignpi_j/dense_3/bias/Adam_1save_5/RestoreV2:24*$
_class
loc:@pi_j/dense_3/bias*
T0*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_5/Assign_25Assignpi_j/dense_3/kernelsave_5/RestoreV2:25*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_j/dense_3/kernel*
use_locking(*
T0
�
save_5/Assign_26Assignpi_j/dense_3/kernel/Adamsave_5/RestoreV2:26*
validate_shape(*
T0*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel*
use_locking(
�
save_5/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_5/RestoreV2:27*
_output_shapes

:*
validate_shape(*
use_locking(*&
_class
loc:@pi_j/dense_3/kernel*
T0
�
save_5/Assign_28Assignpi_n/dense/biassave_5/RestoreV2:28*
T0*
validate_shape(*"
_class
loc:@pi_n/dense/bias*
use_locking(*
_output_shapes
: 
�
save_5/Assign_29Assignpi_n/dense/bias/Adamsave_5/RestoreV2:29*"
_class
loc:@pi_n/dense/bias*
use_locking(*
_output_shapes
: *
T0*
validate_shape(
�
save_5/Assign_30Assignpi_n/dense/bias/Adam_1save_5/RestoreV2:30*
use_locking(*
T0*
validate_shape(*
_output_shapes
: *"
_class
loc:@pi_n/dense/bias
�
save_5/Assign_31Assignpi_n/dense/kernelsave_5/RestoreV2:31*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: 
�
save_5/Assign_32Assignpi_n/dense/kernel/Adamsave_5/RestoreV2:32*
use_locking(*
validate_shape(*$
_class
loc:@pi_n/dense/kernel*
T0*
_output_shapes

: 
�
save_5/Assign_33Assignpi_n/dense/kernel/Adam_1save_5/RestoreV2:33*
validate_shape(*
_output_shapes

: *
T0*
use_locking(*$
_class
loc:@pi_n/dense/kernel
�
save_5/Assign_34Assignpi_n/dense_1/biassave_5/RestoreV2:34*
validate_shape(*
T0*$
_class
loc:@pi_n/dense_1/bias*
use_locking(*
_output_shapes
:
�
save_5/Assign_35Assignpi_n/dense_1/bias/Adamsave_5/RestoreV2:35*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_n/dense_1/bias
�
save_5/Assign_36Assignpi_n/dense_1/bias/Adam_1save_5/RestoreV2:36*$
_class
loc:@pi_n/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
�
save_5/Assign_37Assignpi_n/dense_1/kernelsave_5/RestoreV2:37*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*&
_class
loc:@pi_n/dense_1/kernel
�
save_5/Assign_38Assignpi_n/dense_1/kernel/Adamsave_5/RestoreV2:38*
T0*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
use_locking(*
validate_shape(
�
save_5/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_5/RestoreV2:39*
use_locking(*&
_class
loc:@pi_n/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

: 
�
save_5/Assign_40Assignpi_n/dense_2/biassave_5/RestoreV2:40*$
_class
loc:@pi_n/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_5/Assign_41Assignpi_n/dense_2/bias/Adamsave_5/RestoreV2:41*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
T0*
validate_shape(
�
save_5/Assign_42Assignpi_n/dense_2/bias/Adam_1save_5/RestoreV2:42*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
T0
�
save_5/Assign_43Assignpi_n/dense_2/kernelsave_5/RestoreV2:43*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_2/kernel
�
save_5/Assign_44Assignpi_n/dense_2/kernel/Adamsave_5/RestoreV2:44*
_output_shapes

:*
validate_shape(*
T0*&
_class
loc:@pi_n/dense_2/kernel*
use_locking(
�
save_5/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_5/RestoreV2:45*
validate_shape(*
use_locking(*
T0*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:
�
save_5/Assign_46Assignpi_n/dense_3/biassave_5/RestoreV2:46*
use_locking(*
T0*$
_class
loc:@pi_n/dense_3/bias*
validate_shape(*
_output_shapes
:
�
save_5/Assign_47Assignpi_n/dense_3/bias/Adamsave_5/RestoreV2:47*
validate_shape(*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_n/dense_3/bias*
T0
�
save_5/Assign_48Assignpi_n/dense_3/bias/Adam_1save_5/RestoreV2:48*
use_locking(*
T0*$
_class
loc:@pi_n/dense_3/bias*
_output_shapes
:*
validate_shape(
�
save_5/Assign_49Assignpi_n/dense_3/kernelsave_5/RestoreV2:49*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(*
validate_shape(
�
save_5/Assign_50Assignpi_n/dense_3/kernel/Adamsave_5/RestoreV2:50*
validate_shape(*
_output_shapes

:*
T0*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(
�
save_5/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_5/RestoreV2:51*
use_locking(*&
_class
loc:@pi_n/dense_3/kernel*
validate_shape(*
T0*
_output_shapes

:
�
save_5/Assign_52Assignv/dense/biassave_5/RestoreV2:52*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
T0*
_output_shapes
: 
�
save_5/Assign_53Assignv/dense/bias/Adamsave_5/RestoreV2:53*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_5/Assign_54Assignv/dense/bias/Adam_1save_5/RestoreV2:54*
T0*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(
�
save_5/Assign_55Assignv/dense/kernelsave_5/RestoreV2:55*
validate_shape(*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes
:	� 
�
save_5/Assign_56Assignv/dense/kernel/Adamsave_5/RestoreV2:56*!
_class
loc:@v/dense/kernel*
use_locking(*
T0*
_output_shapes
:	� *
validate_shape(
�
save_5/Assign_57Assignv/dense/kernel/Adam_1save_5/RestoreV2:57*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	� *
use_locking(*
validate_shape(
�
save_5/Assign_58Assignv/dense_1/biassave_5/RestoreV2:58*
validate_shape(*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_1/bias*
T0
�
save_5/Assign_59Assignv/dense_1/bias/Adamsave_5/RestoreV2:59*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
�
save_5/Assign_60Assignv/dense_1/bias/Adam_1save_5/RestoreV2:60*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
T0*
use_locking(
�
save_5/Assign_61Assignv/dense_1/kernelsave_5/RestoreV2:61*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

: 
�
save_5/Assign_62Assignv/dense_1/kernel/Adamsave_5/RestoreV2:62*
T0*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
validate_shape(*
use_locking(
�
save_5/Assign_63Assignv/dense_1/kernel/Adam_1save_5/RestoreV2:63*
T0*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: 
�
save_5/Assign_64Assignv/dense_2/biassave_5/RestoreV2:64*
validate_shape(*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_2/bias*
T0
�
save_5/Assign_65Assignv/dense_2/bias/Adamsave_5/RestoreV2:65*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
save_5/Assign_66Assignv/dense_2/bias/Adam_1save_5/RestoreV2:66*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(
�
save_5/Assign_67Assignv/dense_2/kernelsave_5/RestoreV2:67*#
_class
loc:@v/dense_2/kernel*
use_locking(*
_output_shapes

:*
T0*
validate_shape(
�
save_5/Assign_68Assignv/dense_2/kernel/Adamsave_5/RestoreV2:68*
_output_shapes

:*
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_2/kernel
�
save_5/Assign_69Assignv/dense_2/kernel/Adam_1save_5/RestoreV2:69*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
use_locking(*
validate_shape(*
T0
�
save_5/Assign_70Assignv/dense_3/biassave_5/RestoreV2:70*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*!
_class
loc:@v/dense_3/bias
�
save_5/Assign_71Assignv/dense_3/bias/Adamsave_5/RestoreV2:71*
_output_shapes
:*
validate_shape(*
use_locking(*!
_class
loc:@v/dense_3/bias*
T0
�
save_5/Assign_72Assignv/dense_3/bias/Adam_1save_5/RestoreV2:72*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_3/bias
�
save_5/Assign_73Assignv/dense_3/kernelsave_5/RestoreV2:73*
T0*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
use_locking(
�
save_5/Assign_74Assignv/dense_3/kernel/Adamsave_5/RestoreV2:74*#
_class
loc:@v/dense_3/kernel*
T0*
use_locking(*
_output_shapes

:*
validate_shape(
�
save_5/Assign_75Assignv/dense_3/kernel/Adam_1save_5/RestoreV2:75*
use_locking(*#
_class
loc:@v/dense_3/kernel*
T0*
_output_shapes

:*
validate_shape(
�
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_40^save_5/Assign_41^save_5/Assign_42^save_5/Assign_43^save_5/Assign_44^save_5/Assign_45^save_5/Assign_46^save_5/Assign_47^save_5/Assign_48^save_5/Assign_49^save_5/Assign_5^save_5/Assign_50^save_5/Assign_51^save_5/Assign_52^save_5/Assign_53^save_5/Assign_54^save_5/Assign_55^save_5/Assign_56^save_5/Assign_57^save_5/Assign_58^save_5/Assign_59^save_5/Assign_6^save_5/Assign_60^save_5/Assign_61^save_5/Assign_62^save_5/Assign_63^save_5/Assign_64^save_5/Assign_65^save_5/Assign_66^save_5/Assign_67^save_5/Assign_68^save_5/Assign_69^save_5/Assign_7^save_5/Assign_70^save_5/Assign_71^save_5/Assign_72^save_5/Assign_73^save_5/Assign_74^save_5/Assign_75^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_6/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_de288a775c2342818d1a816e79622d15/part*
dtype0
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
S
save_6/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
�
save_6/SaveV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1
�
save_6/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: 
�
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
_output_shapes
:*
T0*
N*

axis 
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
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1
�
!save_6/RestoreV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:L
�
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L
�
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
use_locking(*
T0*
validate_shape(
�
save_6/Assign_1Assignbeta1_power_1save_6/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
�
save_6/Assign_2Assignbeta2_powersave_6/RestoreV2:2*
use_locking(*
T0*
_output_shapes
: *
validate_shape(*"
_class
loc:@pi_j/dense/bias
�
save_6/Assign_3Assignbeta2_power_1save_6/RestoreV2:3*
T0*
validate_shape(*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: 
�
save_6/Assign_4Assignpi_j/dense/biassave_6/RestoreV2:4*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(*
T0
�
save_6/Assign_5Assignpi_j/dense/bias/Adamsave_6/RestoreV2:5*
_output_shapes
: *
T0*
use_locking(*"
_class
loc:@pi_j/dense/bias*
validate_shape(
�
save_6/Assign_6Assignpi_j/dense/bias/Adam_1save_6/RestoreV2:6*
use_locking(*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
T0*
validate_shape(
�
save_6/Assign_7Assignpi_j/dense/kernelsave_6/RestoreV2:7*$
_class
loc:@pi_j/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

: *
T0
�
save_6/Assign_8Assignpi_j/dense/kernel/Adamsave_6/RestoreV2:8*
validate_shape(*
T0*$
_class
loc:@pi_j/dense/kernel*
use_locking(*
_output_shapes

: 
�
save_6/Assign_9Assignpi_j/dense/kernel/Adam_1save_6/RestoreV2:9*
T0*
_output_shapes

: *
validate_shape(*
use_locking(*$
_class
loc:@pi_j/dense/kernel
�
save_6/Assign_10Assignpi_j/dense_1/biassave_6/RestoreV2:10*$
_class
loc:@pi_j/dense_1/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_6/Assign_11Assignpi_j/dense_1/bias/Adamsave_6/RestoreV2:11*
validate_shape(*
T0*$
_class
loc:@pi_j/dense_1/bias*
use_locking(*
_output_shapes
:
�
save_6/Assign_12Assignpi_j/dense_1/bias/Adam_1save_6/RestoreV2:12*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:
�
save_6/Assign_13Assignpi_j/dense_1/kernelsave_6/RestoreV2:13*
_output_shapes

: *
validate_shape(*&
_class
loc:@pi_j/dense_1/kernel*
T0*
use_locking(
�
save_6/Assign_14Assignpi_j/dense_1/kernel/Adamsave_6/RestoreV2:14*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
T0*
use_locking(*
validate_shape(
�
save_6/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_6/RestoreV2:15*
use_locking(*
validate_shape(*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
T0
�
save_6/Assign_16Assignpi_j/dense_2/biassave_6/RestoreV2:16*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*$
_class
loc:@pi_j/dense_2/bias
�
save_6/Assign_17Assignpi_j/dense_2/bias/Adamsave_6/RestoreV2:17*$
_class
loc:@pi_j/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_6/Assign_18Assignpi_j/dense_2/bias/Adam_1save_6/RestoreV2:18*
T0*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_j/dense_2/bias*
use_locking(
�
save_6/Assign_19Assignpi_j/dense_2/kernelsave_6/RestoreV2:19*
_output_shapes

:*
validate_shape(*
use_locking(*&
_class
loc:@pi_j/dense_2/kernel*
T0
�
save_6/Assign_20Assignpi_j/dense_2/kernel/Adamsave_6/RestoreV2:20*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel*
T0*
_output_shapes

:*
use_locking(
�
save_6/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_6/RestoreV2:21*
_output_shapes

:*
T0*
use_locking(*&
_class
loc:@pi_j/dense_2/kernel*
validate_shape(
�
save_6/Assign_22Assignpi_j/dense_3/biassave_6/RestoreV2:22*
_output_shapes
:*
T0*
use_locking(*$
_class
loc:@pi_j/dense_3/bias*
validate_shape(
�
save_6/Assign_23Assignpi_j/dense_3/bias/Adamsave_6/RestoreV2:23*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*$
_class
loc:@pi_j/dense_3/bias
�
save_6/Assign_24Assignpi_j/dense_3/bias/Adam_1save_6/RestoreV2:24*$
_class
loc:@pi_j/dense_3/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
�
save_6/Assign_25Assignpi_j/dense_3/kernelsave_6/RestoreV2:25*
use_locking(*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel*
validate_shape(*
T0
�
save_6/Assign_26Assignpi_j/dense_3/kernel/Adamsave_6/RestoreV2:26*
T0*&
_class
loc:@pi_j/dense_3/kernel*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_6/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_6/RestoreV2:27*
validate_shape(*
T0*&
_class
loc:@pi_j/dense_3/kernel*
use_locking(*
_output_shapes

:
�
save_6/Assign_28Assignpi_n/dense/biassave_6/RestoreV2:28*
validate_shape(*
use_locking(*"
_class
loc:@pi_n/dense/bias*
T0*
_output_shapes
: 
�
save_6/Assign_29Assignpi_n/dense/bias/Adamsave_6/RestoreV2:29*
use_locking(*
_output_shapes
: *
validate_shape(*
T0*"
_class
loc:@pi_n/dense/bias
�
save_6/Assign_30Assignpi_n/dense/bias/Adam_1save_6/RestoreV2:30*
_output_shapes
: *"
_class
loc:@pi_n/dense/bias*
validate_shape(*
use_locking(*
T0
�
save_6/Assign_31Assignpi_n/dense/kernelsave_6/RestoreV2:31*
_output_shapes

: *
use_locking(*
validate_shape(*$
_class
loc:@pi_n/dense/kernel*
T0
�
save_6/Assign_32Assignpi_n/dense/kernel/Adamsave_6/RestoreV2:32*$
_class
loc:@pi_n/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

: 
�
save_6/Assign_33Assignpi_n/dense/kernel/Adam_1save_6/RestoreV2:33*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: *
use_locking(*
T0*
validate_shape(
�
save_6/Assign_34Assignpi_n/dense_1/biassave_6/RestoreV2:34*
use_locking(*
T0*$
_class
loc:@pi_n/dense_1/bias*
validate_shape(*
_output_shapes
:
�
save_6/Assign_35Assignpi_n/dense_1/bias/Adamsave_6/RestoreV2:35*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:
�
save_6/Assign_36Assignpi_n/dense_1/bias/Adam_1save_6/RestoreV2:36*
_output_shapes
:*$
_class
loc:@pi_n/dense_1/bias*
validate_shape(*
use_locking(*
T0
�
save_6/Assign_37Assignpi_n/dense_1/kernelsave_6/RestoreV2:37*
validate_shape(*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
T0*
use_locking(
�
save_6/Assign_38Assignpi_n/dense_1/kernel/Adamsave_6/RestoreV2:38*
validate_shape(*
use_locking(*
_output_shapes

: *
T0*&
_class
loc:@pi_n/dense_1/kernel
�
save_6/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_6/RestoreV2:39*
_output_shapes

: *
use_locking(*&
_class
loc:@pi_n/dense_1/kernel*
T0*
validate_shape(
�
save_6/Assign_40Assignpi_n/dense_2/biassave_6/RestoreV2:40*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
T0
�
save_6/Assign_41Assignpi_n/dense_2/bias/Adamsave_6/RestoreV2:41*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi_n/dense_2/bias
�
save_6/Assign_42Assignpi_n/dense_2/bias/Adam_1save_6/RestoreV2:42*
T0*$
_class
loc:@pi_n/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_6/Assign_43Assignpi_n/dense_2/kernelsave_6/RestoreV2:43*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*&
_class
loc:@pi_n/dense_2/kernel
�
save_6/Assign_44Assignpi_n/dense_2/kernel/Adamsave_6/RestoreV2:44*
T0*
use_locking(*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:*
validate_shape(
�
save_6/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_6/RestoreV2:45*
_output_shapes

:*
validate_shape(*
use_locking(*&
_class
loc:@pi_n/dense_2/kernel*
T0
�
save_6/Assign_46Assignpi_n/dense_3/biassave_6/RestoreV2:46*
T0*
_output_shapes
:*$
_class
loc:@pi_n/dense_3/bias*
use_locking(*
validate_shape(
�
save_6/Assign_47Assignpi_n/dense_3/bias/Adamsave_6/RestoreV2:47*
validate_shape(*
use_locking(*$
_class
loc:@pi_n/dense_3/bias*
T0*
_output_shapes
:
�
save_6/Assign_48Assignpi_n/dense_3/bias/Adam_1save_6/RestoreV2:48*
validate_shape(*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_n/dense_3/bias*
T0
�
save_6/Assign_49Assignpi_n/dense_3/kernelsave_6/RestoreV2:49*&
_class
loc:@pi_n/dense_3/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

:
�
save_6/Assign_50Assignpi_n/dense_3/kernel/Adamsave_6/RestoreV2:50*
validate_shape(*
use_locking(*&
_class
loc:@pi_n/dense_3/kernel*
_output_shapes

:*
T0
�
save_6/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_6/RestoreV2:51*
T0*
_output_shapes

:*
validate_shape(*
use_locking(*&
_class
loc:@pi_n/dense_3/kernel
�
save_6/Assign_52Assignv/dense/biassave_6/RestoreV2:52*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
�
save_6/Assign_53Assignv/dense/bias/Adamsave_6/RestoreV2:53*
_output_shapes
: *
validate_shape(*
T0*
_class
loc:@v/dense/bias*
use_locking(
�
save_6/Assign_54Assignv/dense/bias/Adam_1save_6/RestoreV2:54*
validate_shape(*
_output_shapes
: *
T0*
use_locking(*
_class
loc:@v/dense/bias
�
save_6/Assign_55Assignv/dense/kernelsave_6/RestoreV2:55*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	� *!
_class
loc:@v/dense/kernel
�
save_6/Assign_56Assignv/dense/kernel/Adamsave_6/RestoreV2:56*
_output_shapes
:	� *
validate_shape(*
T0*
use_locking(*!
_class
loc:@v/dense/kernel
�
save_6/Assign_57Assignv/dense/kernel/Adam_1save_6/RestoreV2:57*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	� *
T0*
use_locking(
�
save_6/Assign_58Assignv/dense_1/biassave_6/RestoreV2:58*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_6/Assign_59Assignv/dense_1/bias/Adamsave_6/RestoreV2:59*
T0*
_output_shapes
:*
use_locking(*!
_class
loc:@v/dense_1/bias*
validate_shape(
�
save_6/Assign_60Assignv/dense_1/bias/Adam_1save_6/RestoreV2:60*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_6/Assign_61Assignv/dense_1/kernelsave_6/RestoreV2:61*
T0*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
use_locking(
�
save_6/Assign_62Assignv/dense_1/kernel/Adamsave_6/RestoreV2:62*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
validate_shape(
�
save_6/Assign_63Assignv/dense_1/kernel/Adam_1save_6/RestoreV2:63*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
T0
�
save_6/Assign_64Assignv/dense_2/biassave_6/RestoreV2:64*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias
�
save_6/Assign_65Assignv/dense_2/bias/Adamsave_6/RestoreV2:65*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
�
save_6/Assign_66Assignv/dense_2/bias/Adam_1save_6/RestoreV2:66*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
�
save_6/Assign_67Assignv/dense_2/kernelsave_6/RestoreV2:67*
validate_shape(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
use_locking(
�
save_6/Assign_68Assignv/dense_2/kernel/Adamsave_6/RestoreV2:68*
_output_shapes

:*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0
�
save_6/Assign_69Assignv/dense_2/kernel/Adam_1save_6/RestoreV2:69*
use_locking(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_6/Assign_70Assignv/dense_3/biassave_6/RestoreV2:70*
_output_shapes
:*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_3/bias
�
save_6/Assign_71Assignv/dense_3/bias/Adamsave_6/RestoreV2:71*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_3/bias
�
save_6/Assign_72Assignv/dense_3/bias/Adam_1save_6/RestoreV2:72*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
use_locking(
�
save_6/Assign_73Assignv/dense_3/kernelsave_6/RestoreV2:73*
validate_shape(*
T0*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
use_locking(
�
save_6/Assign_74Assignv/dense_3/kernel/Adamsave_6/RestoreV2:74*
validate_shape(*
T0*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
use_locking(
�
save_6/Assign_75Assignv/dense_3/kernel/Adam_1save_6/RestoreV2:75*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
_output_shapes

:*
T0*
use_locking(
�
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_40^save_6/Assign_41^save_6/Assign_42^save_6/Assign_43^save_6/Assign_44^save_6/Assign_45^save_6/Assign_46^save_6/Assign_47^save_6/Assign_48^save_6/Assign_49^save_6/Assign_5^save_6/Assign_50^save_6/Assign_51^save_6/Assign_52^save_6/Assign_53^save_6/Assign_54^save_6/Assign_55^save_6/Assign_56^save_6/Assign_57^save_6/Assign_58^save_6/Assign_59^save_6/Assign_6^save_6/Assign_60^save_6/Assign_61^save_6/Assign_62^save_6/Assign_63^save_6/Assign_64^save_6/Assign_65^save_6/Assign_66^save_6/Assign_67^save_6/Assign_68^save_6/Assign_69^save_6/Assign_7^save_6/Assign_70^save_6/Assign_71^save_6/Assign_72^save_6/Assign_73^save_6/Assign_74^save_6/Assign_75^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_c0adc3480e6b48678cab8dc8db29cbcd/part*
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
save_7/num_shardsConst*
value	B :*
_output_shapes
: *
dtype0
^
save_7/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
�
save_7/SaveV2/tensor_namesConst*
_output_shapes
:L*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1
�
save_7/SaveV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
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
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*

axis *
T0*
_output_shapes
:*
N
�
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
�
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
�
save_7/RestoreV2/tensor_namesConst*
dtype0*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
_output_shapes
:L
�
!save_7/RestoreV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:L
�
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L
�
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
_output_shapes
: *
T0*"
_class
loc:@pi_j/dense/bias*
validate_shape(*
use_locking(
�
save_7/Assign_1Assignbeta1_power_1save_7/RestoreV2:1*
use_locking(*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(
�
save_7/Assign_2Assignbeta2_powersave_7/RestoreV2:2*
_output_shapes
: *
validate_shape(*
T0*
use_locking(*"
_class
loc:@pi_j/dense/bias
�
save_7/Assign_3Assignbeta2_power_1save_7/RestoreV2:3*
use_locking(*
validate_shape(*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias
�
save_7/Assign_4Assignpi_j/dense/biassave_7/RestoreV2:4*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*"
_class
loc:@pi_j/dense/bias
�
save_7/Assign_5Assignpi_j/dense/bias/Adamsave_7/RestoreV2:5*"
_class
loc:@pi_j/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
�
save_7/Assign_6Assignpi_j/dense/bias/Adam_1save_7/RestoreV2:6*
validate_shape(*
T0*"
_class
loc:@pi_j/dense/bias*
use_locking(*
_output_shapes
: 
�
save_7/Assign_7Assignpi_j/dense/kernelsave_7/RestoreV2:7*
_output_shapes

: *
T0*$
_class
loc:@pi_j/dense/kernel*
use_locking(*
validate_shape(
�
save_7/Assign_8Assignpi_j/dense/kernel/Adamsave_7/RestoreV2:8*
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel*
validate_shape(*
use_locking(*
T0
�
save_7/Assign_9Assignpi_j/dense/kernel/Adam_1save_7/RestoreV2:9*
_output_shapes

: *
use_locking(*$
_class
loc:@pi_j/dense/kernel*
validate_shape(*
T0
�
save_7/Assign_10Assignpi_j/dense_1/biassave_7/RestoreV2:10*
_output_shapes
:*
use_locking(*
T0*
validate_shape(*$
_class
loc:@pi_j/dense_1/bias
�
save_7/Assign_11Assignpi_j/dense_1/bias/Adamsave_7/RestoreV2:11*
T0*
use_locking(*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_j/dense_1/bias
�
save_7/Assign_12Assignpi_j/dense_1/bias/Adam_1save_7/RestoreV2:12*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:
�
save_7/Assign_13Assignpi_j/dense_1/kernelsave_7/RestoreV2:13*
T0*
_output_shapes

: *
use_locking(*
validate_shape(*&
_class
loc:@pi_j/dense_1/kernel
�
save_7/Assign_14Assignpi_j/dense_1/kernel/Adamsave_7/RestoreV2:14*
T0*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
validate_shape(*
use_locking(
�
save_7/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_7/RestoreV2:15*
validate_shape(*
T0*&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: *
use_locking(
�
save_7/Assign_16Assignpi_j/dense_2/biassave_7/RestoreV2:16*
use_locking(*$
_class
loc:@pi_j/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
�
save_7/Assign_17Assignpi_j/dense_2/bias/Adamsave_7/RestoreV2:17*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_j/dense_2/bias
�
save_7/Assign_18Assignpi_j/dense_2/bias/Adam_1save_7/RestoreV2:18*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:
�
save_7/Assign_19Assignpi_j/dense_2/kernelsave_7/RestoreV2:19*
validate_shape(*
use_locking(*&
_class
loc:@pi_j/dense_2/kernel*
_output_shapes

:*
T0
�
save_7/Assign_20Assignpi_j/dense_2/kernel/Adamsave_7/RestoreV2:20*
validate_shape(*
T0*&
_class
loc:@pi_j/dense_2/kernel*
use_locking(*
_output_shapes

:
�
save_7/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_7/RestoreV2:21*
_output_shapes

:*
T0*
use_locking(*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel
�
save_7/Assign_22Assignpi_j/dense_3/biassave_7/RestoreV2:22*
validate_shape(*
T0*
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias*
use_locking(
�
save_7/Assign_23Assignpi_j/dense_3/bias/Adamsave_7/RestoreV2:23*
use_locking(*$
_class
loc:@pi_j/dense_3/bias*
_output_shapes
:*
validate_shape(*
T0
�
save_7/Assign_24Assignpi_j/dense_3/bias/Adam_1save_7/RestoreV2:24*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*$
_class
loc:@pi_j/dense_3/bias
�
save_7/Assign_25Assignpi_j/dense_3/kernelsave_7/RestoreV2:25*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:*
validate_shape(*
use_locking(*
T0
�
save_7/Assign_26Assignpi_j/dense_3/kernel/Adamsave_7/RestoreV2:26*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*&
_class
loc:@pi_j/dense_3/kernel
�
save_7/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_7/RestoreV2:27*
T0*
_output_shapes

:*
use_locking(*
validate_shape(*&
_class
loc:@pi_j/dense_3/kernel
�
save_7/Assign_28Assignpi_n/dense/biassave_7/RestoreV2:28*
_output_shapes
: *
T0*
use_locking(*"
_class
loc:@pi_n/dense/bias*
validate_shape(
�
save_7/Assign_29Assignpi_n/dense/bias/Adamsave_7/RestoreV2:29*
_output_shapes
: *"
_class
loc:@pi_n/dense/bias*
use_locking(*
T0*
validate_shape(
�
save_7/Assign_30Assignpi_n/dense/bias/Adam_1save_7/RestoreV2:30*
T0*
validate_shape(*
use_locking(*"
_class
loc:@pi_n/dense/bias*
_output_shapes
: 
�
save_7/Assign_31Assignpi_n/dense/kernelsave_7/RestoreV2:31*$
_class
loc:@pi_n/dense/kernel*
T0*
validate_shape(*
_output_shapes

: *
use_locking(
�
save_7/Assign_32Assignpi_n/dense/kernel/Adamsave_7/RestoreV2:32*
use_locking(*
T0*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel*
validate_shape(
�
save_7/Assign_33Assignpi_n/dense/kernel/Adam_1save_7/RestoreV2:33*
validate_shape(*
_output_shapes

: *
T0*$
_class
loc:@pi_n/dense/kernel*
use_locking(
�
save_7/Assign_34Assignpi_n/dense_1/biassave_7/RestoreV2:34*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:
�
save_7/Assign_35Assignpi_n/dense_1/bias/Adamsave_7/RestoreV2:35*
T0*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(
�
save_7/Assign_36Assignpi_n/dense_1/bias/Adam_1save_7/RestoreV2:36*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi_n/dense_1/bias
�
save_7/Assign_37Assignpi_n/dense_1/kernelsave_7/RestoreV2:37*
validate_shape(*&
_class
loc:@pi_n/dense_1/kernel*
use_locking(*
_output_shapes

: *
T0
�
save_7/Assign_38Assignpi_n/dense_1/kernel/Adamsave_7/RestoreV2:38*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
use_locking(*
validate_shape(*
T0
�
save_7/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_7/RestoreV2:39*
validate_shape(*
_output_shapes

: *
T0*&
_class
loc:@pi_n/dense_1/kernel*
use_locking(
�
save_7/Assign_40Assignpi_n/dense_2/biassave_7/RestoreV2:40*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_7/Assign_41Assignpi_n/dense_2/bias/Adamsave_7/RestoreV2:41*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
�
save_7/Assign_42Assignpi_n/dense_2/bias/Adam_1save_7/RestoreV2:42*
use_locking(*$
_class
loc:@pi_n/dense_2/bias*
T0*
_output_shapes
:*
validate_shape(
�
save_7/Assign_43Assignpi_n/dense_2/kernelsave_7/RestoreV2:43*
_output_shapes

:*
use_locking(*&
_class
loc:@pi_n/dense_2/kernel*
validate_shape(*
T0
�
save_7/Assign_44Assignpi_n/dense_2/kernel/Adamsave_7/RestoreV2:44*
_output_shapes

:*
T0*
use_locking(*
validate_shape(*&
_class
loc:@pi_n/dense_2/kernel
�
save_7/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_7/RestoreV2:45*
validate_shape(*
_output_shapes

:*
T0*&
_class
loc:@pi_n/dense_2/kernel*
use_locking(
�
save_7/Assign_46Assignpi_n/dense_3/biassave_7/RestoreV2:46*
_output_shapes
:*
T0*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
use_locking(
�
save_7/Assign_47Assignpi_n/dense_3/bias/Adamsave_7/RestoreV2:47*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
use_locking(*
T0*
_output_shapes
:
�
save_7/Assign_48Assignpi_n/dense_3/bias/Adam_1save_7/RestoreV2:48*
use_locking(*$
_class
loc:@pi_n/dense_3/bias*
_output_shapes
:*
validate_shape(*
T0
�
save_7/Assign_49Assignpi_n/dense_3/kernelsave_7/RestoreV2:49*
use_locking(*
_output_shapes

:*
validate_shape(*
T0*&
_class
loc:@pi_n/dense_3/kernel
�
save_7/Assign_50Assignpi_n/dense_3/kernel/Adamsave_7/RestoreV2:50*
T0*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(*
validate_shape(*
_output_shapes

:
�
save_7/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_7/RestoreV2:51*
use_locking(*
_output_shapes

:*
validate_shape(*
T0*&
_class
loc:@pi_n/dense_3/kernel
�
save_7/Assign_52Assignv/dense/biassave_7/RestoreV2:52*
_class
loc:@v/dense/bias*
T0*
use_locking(*
_output_shapes
: *
validate_shape(
�
save_7/Assign_53Assignv/dense/bias/Adamsave_7/RestoreV2:53*
_class
loc:@v/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
�
save_7/Assign_54Assignv/dense/bias/Adam_1save_7/RestoreV2:54*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
validate_shape(*
use_locking(
�
save_7/Assign_55Assignv/dense/kernelsave_7/RestoreV2:55*
validate_shape(*!
_class
loc:@v/dense/kernel*
T0*
_output_shapes
:	� *
use_locking(
�
save_7/Assign_56Assignv/dense/kernel/Adamsave_7/RestoreV2:56*
_output_shapes
:	� *!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
use_locking(
�
save_7/Assign_57Assignv/dense/kernel/Adam_1save_7/RestoreV2:57*!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
_output_shapes
:	� *
use_locking(
�
save_7/Assign_58Assignv/dense_1/biassave_7/RestoreV2:58*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_1/bias
�
save_7/Assign_59Assignv/dense_1/bias/Adamsave_7/RestoreV2:59*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_1/bias
�
save_7/Assign_60Assignv/dense_1/bias/Adam_1save_7/RestoreV2:60*
T0*!
_class
loc:@v/dense_1/bias*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_7/Assign_61Assignv/dense_1/kernelsave_7/RestoreV2:61*#
_class
loc:@v/dense_1/kernel*
use_locking(*
_output_shapes

: *
T0*
validate_shape(
�
save_7/Assign_62Assignv/dense_1/kernel/Adamsave_7/RestoreV2:62*
use_locking(*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
T0
�
save_7/Assign_63Assignv/dense_1/kernel/Adam_1save_7/RestoreV2:63*
use_locking(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0
�
save_7/Assign_64Assignv/dense_2/biassave_7/RestoreV2:64*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
use_locking(
�
save_7/Assign_65Assignv/dense_2/bias/Adamsave_7/RestoreV2:65*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_7/Assign_66Assignv/dense_2/bias/Adam_1save_7/RestoreV2:66*!
_class
loc:@v/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
�
save_7/Assign_67Assignv/dense_2/kernelsave_7/RestoreV2:67*
T0*
validate_shape(*
_output_shapes

:*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
save_7/Assign_68Assignv/dense_2/kernel/Adamsave_7/RestoreV2:68*
_output_shapes

:*
validate_shape(*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel
�
save_7/Assign_69Assignv/dense_2/kernel/Adam_1save_7/RestoreV2:69*
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:
�
save_7/Assign_70Assignv/dense_3/biassave_7/RestoreV2:70*
validate_shape(*
use_locking(*
T0*!
_class
loc:@v/dense_3/bias*
_output_shapes
:
�
save_7/Assign_71Assignv/dense_3/bias/Adamsave_7/RestoreV2:71*
T0*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
validate_shape(*
use_locking(
�
save_7/Assign_72Assignv/dense_3/bias/Adam_1save_7/RestoreV2:72*
validate_shape(*
T0*!
_class
loc:@v/dense_3/bias*
_output_shapes
:*
use_locking(
�
save_7/Assign_73Assignv/dense_3/kernelsave_7/RestoreV2:73*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
use_locking(*
T0
�
save_7/Assign_74Assignv/dense_3/kernel/Adamsave_7/RestoreV2:74*
T0*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
use_locking(*
_output_shapes

:
�
save_7/Assign_75Assignv/dense_3/kernel/Adam_1save_7/RestoreV2:75*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
use_locking(*
T0
�
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_40^save_7/Assign_41^save_7/Assign_42^save_7/Assign_43^save_7/Assign_44^save_7/Assign_45^save_7/Assign_46^save_7/Assign_47^save_7/Assign_48^save_7/Assign_49^save_7/Assign_5^save_7/Assign_50^save_7/Assign_51^save_7/Assign_52^save_7/Assign_53^save_7/Assign_54^save_7/Assign_55^save_7/Assign_56^save_7/Assign_57^save_7/Assign_58^save_7/Assign_59^save_7/Assign_6^save_7/Assign_60^save_7/Assign_61^save_7/Assign_62^save_7/Assign_63^save_7/Assign_64^save_7/Assign_65^save_7/Assign_66^save_7/Assign_67^save_7/Assign_68^save_7/Assign_69^save_7/Assign_7^save_7/Assign_70^save_7/Assign_71^save_7/Assign_72^save_7/Assign_73^save_7/Assign_74^save_7/Assign_75^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
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
dtype0*
shape: 
�
save_8/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_87cba67813014b649352c3799b16bb8e/part*
dtype0
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
	separator *
_output_shapes
: *
N
S
save_8/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
^
save_8/ShardedFilename/shardConst*
dtype0*
value	B : *
_output_shapes
: 
�
save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
�
save_8/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1
�
save_8/SaveV2/shape_and_slicesConst*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L*
dtype0
�
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
_output_shapes
: *)
_class
loc:@save_8/ShardedFilename*
T0
�
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
N*

axis *
T0*
_output_shapes
:
�
save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
�
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
T0*
_output_shapes
: 
�
save_8/RestoreV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0
�
!save_8/RestoreV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
_output_shapes
: *
T0*"
_class
loc:@pi_j/dense/bias*
use_locking(*
validate_shape(
�
save_8/Assign_1Assignbeta1_power_1save_8/RestoreV2:1*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(*
T0
�
save_8/Assign_2Assignbeta2_powersave_8/RestoreV2:2*
T0*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
�
save_8/Assign_3Assignbeta2_power_1save_8/RestoreV2:3*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
�
save_8/Assign_4Assignpi_j/dense/biassave_8/RestoreV2:4*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
use_locking(*
T0*
validate_shape(
�
save_8/Assign_5Assignpi_j/dense/bias/Adamsave_8/RestoreV2:5*
T0*
_output_shapes
: *
validate_shape(*
use_locking(*"
_class
loc:@pi_j/dense/bias
�
save_8/Assign_6Assignpi_j/dense/bias/Adam_1save_8/RestoreV2:6*
T0*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: *
use_locking(*
validate_shape(
�
save_8/Assign_7Assignpi_j/dense/kernelsave_8/RestoreV2:7*
use_locking(*
T0*$
_class
loc:@pi_j/dense/kernel*
validate_shape(*
_output_shapes

: 
�
save_8/Assign_8Assignpi_j/dense/kernel/Adamsave_8/RestoreV2:8*
validate_shape(*
T0*
_output_shapes

: *
use_locking(*$
_class
loc:@pi_j/dense/kernel
�
save_8/Assign_9Assignpi_j/dense/kernel/Adam_1save_8/RestoreV2:9*
validate_shape(*
use_locking(*
_output_shapes

: *$
_class
loc:@pi_j/dense/kernel*
T0
�
save_8/Assign_10Assignpi_j/dense_1/biassave_8/RestoreV2:10*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_8/Assign_11Assignpi_j/dense_1/bias/Adamsave_8/RestoreV2:11*$
_class
loc:@pi_j/dense_1/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
:
�
save_8/Assign_12Assignpi_j/dense_1/bias/Adam_1save_8/RestoreV2:12*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi_j/dense_1/bias*
_output_shapes
:
�
save_8/Assign_13Assignpi_j/dense_1/kernelsave_8/RestoreV2:13*&
_class
loc:@pi_j/dense_1/kernel*
use_locking(*
_output_shapes

: *
T0*
validate_shape(
�
save_8/Assign_14Assignpi_j/dense_1/kernel/Adamsave_8/RestoreV2:14*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
T0*
use_locking(*
validate_shape(
�
save_8/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_8/RestoreV2:15*&
_class
loc:@pi_j/dense_1/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

: 
�
save_8/Assign_16Assignpi_j/dense_2/biassave_8/RestoreV2:16*
use_locking(*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_j/dense_2/bias*
T0
�
save_8/Assign_17Assignpi_j/dense_2/bias/Adamsave_8/RestoreV2:17*
_output_shapes
:*
T0*
use_locking(*$
_class
loc:@pi_j/dense_2/bias*
validate_shape(
�
save_8/Assign_18Assignpi_j/dense_2/bias/Adam_1save_8/RestoreV2:18*
use_locking(*$
_class
loc:@pi_j/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(
�
save_8/Assign_19Assignpi_j/dense_2/kernelsave_8/RestoreV2:19*
use_locking(*
_output_shapes

:*
T0*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel
�
save_8/Assign_20Assignpi_j/dense_2/kernel/Adamsave_8/RestoreV2:20*
T0*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel*
use_locking(
�
save_8/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_8/RestoreV2:21*
_output_shapes

:*
T0*
use_locking(*&
_class
loc:@pi_j/dense_2/kernel*
validate_shape(
�
save_8/Assign_22Assignpi_j/dense_3/biassave_8/RestoreV2:22*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_j/dense_3/bias*
T0*
validate_shape(
�
save_8/Assign_23Assignpi_j/dense_3/bias/Adamsave_8/RestoreV2:23*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@pi_j/dense_3/bias*
validate_shape(
�
save_8/Assign_24Assignpi_j/dense_3/bias/Adam_1save_8/RestoreV2:24*$
_class
loc:@pi_j/dense_3/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
�
save_8/Assign_25Assignpi_j/dense_3/kernelsave_8/RestoreV2:25*
use_locking(*
_output_shapes

:*
validate_shape(*
T0*&
_class
loc:@pi_j/dense_3/kernel
�
save_8/Assign_26Assignpi_j/dense_3/kernel/Adamsave_8/RestoreV2:26*
validate_shape(*
use_locking(*
T0*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel
�
save_8/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_8/RestoreV2:27*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel*
T0*
use_locking(*
validate_shape(
�
save_8/Assign_28Assignpi_n/dense/biassave_8/RestoreV2:28*"
_class
loc:@pi_n/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
�
save_8/Assign_29Assignpi_n/dense/bias/Adamsave_8/RestoreV2:29*
T0*"
_class
loc:@pi_n/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_8/Assign_30Assignpi_n/dense/bias/Adam_1save_8/RestoreV2:30*
validate_shape(*"
_class
loc:@pi_n/dense/bias*
_output_shapes
: *
T0*
use_locking(
�
save_8/Assign_31Assignpi_n/dense/kernelsave_8/RestoreV2:31*
T0*
use_locking(*$
_class
loc:@pi_n/dense/kernel*
validate_shape(*
_output_shapes

: 
�
save_8/Assign_32Assignpi_n/dense/kernel/Adamsave_8/RestoreV2:32*
T0*$
_class
loc:@pi_n/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes

: 
�
save_8/Assign_33Assignpi_n/dense/kernel/Adam_1save_8/RestoreV2:33*
use_locking(*
T0*
_output_shapes

: *
validate_shape(*$
_class
loc:@pi_n/dense/kernel
�
save_8/Assign_34Assignpi_n/dense_1/biassave_8/RestoreV2:34*
use_locking(*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_n/dense_1/bias*
T0
�
save_8/Assign_35Assignpi_n/dense_1/bias/Adamsave_8/RestoreV2:35*
use_locking(*
T0*
_output_shapes
:*$
_class
loc:@pi_n/dense_1/bias*
validate_shape(
�
save_8/Assign_36Assignpi_n/dense_1/bias/Adam_1save_8/RestoreV2:36*
use_locking(*
validate_shape(*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:*
T0
�
save_8/Assign_37Assignpi_n/dense_1/kernelsave_8/RestoreV2:37*&
_class
loc:@pi_n/dense_1/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

: 
�
save_8/Assign_38Assignpi_n/dense_1/kernel/Adamsave_8/RestoreV2:38*
validate_shape(*&
_class
loc:@pi_n/dense_1/kernel*
use_locking(*
T0*
_output_shapes

: 
�
save_8/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_8/RestoreV2:39*
validate_shape(*
use_locking(*
_output_shapes

: *
T0*&
_class
loc:@pi_n/dense_1/kernel
�
save_8/Assign_40Assignpi_n/dense_2/biassave_8/RestoreV2:40*
validate_shape(*
T0*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
_output_shapes
:
�
save_8/Assign_41Assignpi_n/dense_2/bias/Adamsave_8/RestoreV2:41*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
use_locking(
�
save_8/Assign_42Assignpi_n/dense_2/bias/Adam_1save_8/RestoreV2:42*$
_class
loc:@pi_n/dense_2/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_8/Assign_43Assignpi_n/dense_2/kernelsave_8/RestoreV2:43*&
_class
loc:@pi_n/dense_2/kernel*
validate_shape(*
T0*
_output_shapes

:*
use_locking(
�
save_8/Assign_44Assignpi_n/dense_2/kernel/Adamsave_8/RestoreV2:44*&
_class
loc:@pi_n/dense_2/kernel*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
save_8/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_8/RestoreV2:45*&
_class
loc:@pi_n/dense_2/kernel*
T0*
_output_shapes

:*
validate_shape(*
use_locking(
�
save_8/Assign_46Assignpi_n/dense_3/biassave_8/RestoreV2:46*$
_class
loc:@pi_n/dense_3/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(
�
save_8/Assign_47Assignpi_n/dense_3/bias/Adamsave_8/RestoreV2:47*
T0*$
_class
loc:@pi_n/dense_3/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_8/Assign_48Assignpi_n/dense_3/bias/Adam_1save_8/RestoreV2:48*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_3/bias*
use_locking(*
validate_shape(
�
save_8/Assign_49Assignpi_n/dense_3/kernelsave_8/RestoreV2:49*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(*
validate_shape(*
_output_shapes

:*
T0
�
save_8/Assign_50Assignpi_n/dense_3/kernel/Adamsave_8/RestoreV2:50*
T0*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(*
validate_shape(
�
save_8/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_8/RestoreV2:51*
use_locking(*
_output_shapes

:*&
_class
loc:@pi_n/dense_3/kernel*
validate_shape(*
T0
�
save_8/Assign_52Assignv/dense/biassave_8/RestoreV2:52*
use_locking(*
T0*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(
�
save_8/Assign_53Assignv/dense/bias/Adamsave_8/RestoreV2:53*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_8/Assign_54Assignv/dense/bias/Adam_1save_8/RestoreV2:54*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
validate_shape(*
use_locking(
�
save_8/Assign_55Assignv/dense/kernelsave_8/RestoreV2:55*
_output_shapes
:	� *
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
validate_shape(
�
save_8/Assign_56Assignv/dense/kernel/Adamsave_8/RestoreV2:56*
T0*
_output_shapes
:	� *
use_locking(*
validate_shape(*!
_class
loc:@v/dense/kernel
�
save_8/Assign_57Assignv/dense/kernel/Adam_1save_8/RestoreV2:57*
_output_shapes
:	� *!
_class
loc:@v/dense/kernel*
use_locking(*
T0*
validate_shape(
�
save_8/Assign_58Assignv/dense_1/biassave_8/RestoreV2:58*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_1/bias*
use_locking(*
T0
�
save_8/Assign_59Assignv/dense_1/bias/Adamsave_8/RestoreV2:59*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias
�
save_8/Assign_60Assignv/dense_1/bias/Adam_1save_8/RestoreV2:60*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*!
_class
loc:@v/dense_1/bias
�
save_8/Assign_61Assignv/dense_1/kernelsave_8/RestoreV2:61*
_output_shapes

: *
validate_shape(*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
�
save_8/Assign_62Assignv/dense_1/kernel/Adamsave_8/RestoreV2:62*#
_class
loc:@v/dense_1/kernel*
T0*
use_locking(*
_output_shapes

: *
validate_shape(
�
save_8/Assign_63Assignv/dense_1/kernel/Adam_1save_8/RestoreV2:63*
validate_shape(*
_output_shapes

: *
T0*
use_locking(*#
_class
loc:@v/dense_1/kernel
�
save_8/Assign_64Assignv/dense_2/biassave_8/RestoreV2:64*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*!
_class
loc:@v/dense_2/bias
�
save_8/Assign_65Assignv/dense_2/bias/Adamsave_8/RestoreV2:65*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_2/bias*
T0*
use_locking(
�
save_8/Assign_66Assignv/dense_2/bias/Adam_1save_8/RestoreV2:66*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_8/Assign_67Assignv/dense_2/kernelsave_8/RestoreV2:67*
validate_shape(*
use_locking(*
T0*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel
�
save_8/Assign_68Assignv/dense_2/kernel/Adamsave_8/RestoreV2:68*
use_locking(*
validate_shape(*#
_class
loc:@v/dense_2/kernel*
T0*
_output_shapes

:
�
save_8/Assign_69Assignv/dense_2/kernel/Adam_1save_8/RestoreV2:69*#
_class
loc:@v/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

:
�
save_8/Assign_70Assignv/dense_3/biassave_8/RestoreV2:70*
use_locking(*
validate_shape(*
T0*!
_class
loc:@v/dense_3/bias*
_output_shapes
:
�
save_8/Assign_71Assignv/dense_3/bias/Adamsave_8/RestoreV2:71*
validate_shape(*!
_class
loc:@v/dense_3/bias*
T0*
use_locking(*
_output_shapes
:
�
save_8/Assign_72Assignv/dense_3/bias/Adam_1save_8/RestoreV2:72*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_3/bias*
T0*
use_locking(
�
save_8/Assign_73Assignv/dense_3/kernelsave_8/RestoreV2:73*
validate_shape(*
T0*#
_class
loc:@v/dense_3/kernel*
use_locking(*
_output_shapes

:
�
save_8/Assign_74Assignv/dense_3/kernel/Adamsave_8/RestoreV2:74*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0*
use_locking(
�
save_8/Assign_75Assignv/dense_3/kernel/Adam_1save_8/RestoreV2:75*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
T0*
use_locking(*
validate_shape(
�
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_40^save_8/Assign_41^save_8/Assign_42^save_8/Assign_43^save_8/Assign_44^save_8/Assign_45^save_8/Assign_46^save_8/Assign_47^save_8/Assign_48^save_8/Assign_49^save_8/Assign_5^save_8/Assign_50^save_8/Assign_51^save_8/Assign_52^save_8/Assign_53^save_8/Assign_54^save_8/Assign_55^save_8/Assign_56^save_8/Assign_57^save_8/Assign_58^save_8/Assign_59^save_8/Assign_6^save_8/Assign_60^save_8/Assign_61^save_8/Assign_62^save_8/Assign_63^save_8/Assign_64^save_8/Assign_65^save_8/Assign_66^save_8/Assign_67^save_8/Assign_68^save_8/Assign_69^save_8/Assign_7^save_8/Assign_70^save_8/Assign_71^save_8/Assign_72^save_8/Assign_73^save_8/Assign_74^save_8/Assign_75^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
shape: *
_output_shapes
: *
dtype0
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
_output_shapes
: *
dtype0*
shape: 
�
save_9/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_c4cc612b201f461b8b56826e090acd59/part
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_9/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
�
save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
�
save_9/SaveV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0
�
save_9/SaveV2/shape_and_slicesConst*
_output_shapes
:L*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_9/ShardedFilename
�
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
_output_shapes
:*
T0*

axis *
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
save_9/RestoreV2/tensor_namesConst*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0*
_output_shapes
:L
�
!save_9/RestoreV2/shape_and_slicesConst*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
T0*"
_class
loc:@pi_j/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: 
�
save_9/Assign_1Assignbeta1_power_1save_9/RestoreV2:1*
validate_shape(*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
use_locking(
�
save_9/Assign_2Assignbeta2_powersave_9/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@pi_j/dense/bias*
validate_shape(
�
save_9/Assign_3Assignbeta2_power_1save_9/RestoreV2:3*
_class
loc:@v/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
�
save_9/Assign_4Assignpi_j/dense/biassave_9/RestoreV2:4*
_output_shapes
: *
validate_shape(*
use_locking(*"
_class
loc:@pi_j/dense/bias*
T0
�
save_9/Assign_5Assignpi_j/dense/bias/Adamsave_9/RestoreV2:5*
T0*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
validate_shape(*
use_locking(
�
save_9/Assign_6Assignpi_j/dense/bias/Adam_1save_9/RestoreV2:6*
use_locking(*"
_class
loc:@pi_j/dense/bias*
T0*
_output_shapes
: *
validate_shape(
�
save_9/Assign_7Assignpi_j/dense/kernelsave_9/RestoreV2:7*
validate_shape(*$
_class
loc:@pi_j/dense/kernel*
T0*
use_locking(*
_output_shapes

: 
�
save_9/Assign_8Assignpi_j/dense/kernel/Adamsave_9/RestoreV2:8*
validate_shape(*$
_class
loc:@pi_j/dense/kernel*
use_locking(*
_output_shapes

: *
T0
�
save_9/Assign_9Assignpi_j/dense/kernel/Adam_1save_9/RestoreV2:9*
validate_shape(*$
_class
loc:@pi_j/dense/kernel*
_output_shapes

: *
T0*
use_locking(
�
save_9/Assign_10Assignpi_j/dense_1/biassave_9/RestoreV2:10*
T0*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_j/dense_1/bias*
validate_shape(
�
save_9/Assign_11Assignpi_j/dense_1/bias/Adamsave_9/RestoreV2:11*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_j/dense_1/bias
�
save_9/Assign_12Assignpi_j/dense_1/bias/Adam_1save_9/RestoreV2:12*
validate_shape(*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_j/dense_1/bias*
T0
�
save_9/Assign_13Assignpi_j/dense_1/kernelsave_9/RestoreV2:13*
T0*
validate_shape(*&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: *
use_locking(
�
save_9/Assign_14Assignpi_j/dense_1/kernel/Adamsave_9/RestoreV2:14*
T0*
validate_shape(*&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: *
use_locking(
�
save_9/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_9/RestoreV2:15*
validate_shape(*
_output_shapes

: *
use_locking(*&
_class
loc:@pi_j/dense_1/kernel*
T0
�
save_9/Assign_16Assignpi_j/dense_2/biassave_9/RestoreV2:16*
T0*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_j/dense_2/bias*
validate_shape(
�
save_9/Assign_17Assignpi_j/dense_2/bias/Adamsave_9/RestoreV2:17*
_output_shapes
:*
T0*$
_class
loc:@pi_j/dense_2/bias*
validate_shape(*
use_locking(
�
save_9/Assign_18Assignpi_j/dense_2/bias/Adam_1save_9/RestoreV2:18*$
_class
loc:@pi_j/dense_2/bias*
T0*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_9/Assign_19Assignpi_j/dense_2/kernelsave_9/RestoreV2:19*
validate_shape(*
_output_shapes

:*
use_locking(*&
_class
loc:@pi_j/dense_2/kernel*
T0
�
save_9/Assign_20Assignpi_j/dense_2/kernel/Adamsave_9/RestoreV2:20*
T0*&
_class
loc:@pi_j/dense_2/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
save_9/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_9/RestoreV2:21*
T0*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel*
use_locking(*
_output_shapes

:
�
save_9/Assign_22Assignpi_j/dense_3/biassave_9/RestoreV2:22*
_output_shapes
:*
use_locking(*
validate_shape(*$
_class
loc:@pi_j/dense_3/bias*
T0
�
save_9/Assign_23Assignpi_j/dense_3/bias/Adamsave_9/RestoreV2:23*
validate_shape(*
T0*
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias*
use_locking(
�
save_9/Assign_24Assignpi_j/dense_3/bias/Adam_1save_9/RestoreV2:24*
T0*
_output_shapes
:*$
_class
loc:@pi_j/dense_3/bias*
use_locking(*
validate_shape(
�
save_9/Assign_25Assignpi_j/dense_3/kernelsave_9/RestoreV2:25*
validate_shape(*&
_class
loc:@pi_j/dense_3/kernel*
use_locking(*
_output_shapes

:*
T0
�
save_9/Assign_26Assignpi_j/dense_3/kernel/Adamsave_9/RestoreV2:26*&
_class
loc:@pi_j/dense_3/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:
�
save_9/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_9/RestoreV2:27*
validate_shape(*&
_class
loc:@pi_j/dense_3/kernel*
_output_shapes

:*
use_locking(*
T0
�
save_9/Assign_28Assignpi_n/dense/biassave_9/RestoreV2:28*
use_locking(*
_output_shapes
: *
validate_shape(*
T0*"
_class
loc:@pi_n/dense/bias
�
save_9/Assign_29Assignpi_n/dense/bias/Adamsave_9/RestoreV2:29*
validate_shape(*
_output_shapes
: *"
_class
loc:@pi_n/dense/bias*
T0*
use_locking(
�
save_9/Assign_30Assignpi_n/dense/bias/Adam_1save_9/RestoreV2:30*
_output_shapes
: *
use_locking(*
validate_shape(*"
_class
loc:@pi_n/dense/bias*
T0
�
save_9/Assign_31Assignpi_n/dense/kernelsave_9/RestoreV2:31*
_output_shapes

: *
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi_n/dense/kernel
�
save_9/Assign_32Assignpi_n/dense/kernel/Adamsave_9/RestoreV2:32*
T0*
_output_shapes

: *
use_locking(*$
_class
loc:@pi_n/dense/kernel*
validate_shape(
�
save_9/Assign_33Assignpi_n/dense/kernel/Adam_1save_9/RestoreV2:33*
T0*
use_locking(*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel*
validate_shape(
�
save_9/Assign_34Assignpi_n/dense_1/biassave_9/RestoreV2:34*
validate_shape(*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:*
use_locking(*
T0
�
save_9/Assign_35Assignpi_n/dense_1/bias/Adamsave_9/RestoreV2:35*
_output_shapes
:*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi_n/dense_1/bias
�
save_9/Assign_36Assignpi_n/dense_1/bias/Adam_1save_9/RestoreV2:36*
use_locking(*
T0*$
_class
loc:@pi_n/dense_1/bias*
validate_shape(*
_output_shapes
:
�
save_9/Assign_37Assignpi_n/dense_1/kernelsave_9/RestoreV2:37*
_output_shapes

: *
T0*
validate_shape(*&
_class
loc:@pi_n/dense_1/kernel*
use_locking(
�
save_9/Assign_38Assignpi_n/dense_1/kernel/Adamsave_9/RestoreV2:38*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
T0*
use_locking(*
validate_shape(
�
save_9/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_9/RestoreV2:39*&
_class
loc:@pi_n/dense_1/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
�
save_9/Assign_40Assignpi_n/dense_2/biassave_9/RestoreV2:40*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_n/dense_2/bias*
validate_shape(*
T0
�
save_9/Assign_41Assignpi_n/dense_2/bias/Adamsave_9/RestoreV2:41*
validate_shape(*
_output_shapes
:*
T0*
use_locking(*$
_class
loc:@pi_n/dense_2/bias
�
save_9/Assign_42Assignpi_n/dense_2/bias/Adam_1save_9/RestoreV2:42*
T0*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
validate_shape(
�
save_9/Assign_43Assignpi_n/dense_2/kernelsave_9/RestoreV2:43*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:*
use_locking(*
validate_shape(*
T0
�
save_9/Assign_44Assignpi_n/dense_2/kernel/Adamsave_9/RestoreV2:44*
T0*&
_class
loc:@pi_n/dense_2/kernel*
_output_shapes

:*
validate_shape(*
use_locking(
�
save_9/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_9/RestoreV2:45*&
_class
loc:@pi_n/dense_2/kernel*
use_locking(*
validate_shape(*
_output_shapes

:*
T0
�
save_9/Assign_46Assignpi_n/dense_3/biassave_9/RestoreV2:46*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*$
_class
loc:@pi_n/dense_3/bias
�
save_9/Assign_47Assignpi_n/dense_3/bias/Adamsave_9/RestoreV2:47*
_output_shapes
:*
use_locking(*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
T0
�
save_9/Assign_48Assignpi_n/dense_3/bias/Adam_1save_9/RestoreV2:48*$
_class
loc:@pi_n/dense_3/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
�
save_9/Assign_49Assignpi_n/dense_3/kernelsave_9/RestoreV2:49*
T0*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(*
_output_shapes

:*
validate_shape(
�
save_9/Assign_50Assignpi_n/dense_3/kernel/Adamsave_9/RestoreV2:50*&
_class
loc:@pi_n/dense_3/kernel*
use_locking(*
T0*
_output_shapes

:*
validate_shape(
�
save_9/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_9/RestoreV2:51*
_output_shapes

:*
use_locking(*
T0*
validate_shape(*&
_class
loc:@pi_n/dense_3/kernel
�
save_9/Assign_52Assignv/dense/biassave_9/RestoreV2:52*
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
T0
�
save_9/Assign_53Assignv/dense/bias/Adamsave_9/RestoreV2:53*
use_locking(*
_output_shapes
: *
_class
loc:@v/dense/bias*
validate_shape(*
T0
�
save_9/Assign_54Assignv/dense/bias/Adam_1save_9/RestoreV2:54*
T0*
_output_shapes
: *
use_locking(*
_class
loc:@v/dense/bias*
validate_shape(
�
save_9/Assign_55Assignv/dense/kernelsave_9/RestoreV2:55*
_output_shapes
:	� *!
_class
loc:@v/dense/kernel*
validate_shape(*
T0*
use_locking(
�
save_9/Assign_56Assignv/dense/kernel/Adamsave_9/RestoreV2:56*
T0*
use_locking(*!
_class
loc:@v/dense/kernel*
_output_shapes
:	� *
validate_shape(
�
save_9/Assign_57Assignv/dense/kernel/Adam_1save_9/RestoreV2:57*
T0*!
_class
loc:@v/dense/kernel*
use_locking(*
_output_shapes
:	� *
validate_shape(
�
save_9/Assign_58Assignv/dense_1/biassave_9/RestoreV2:58*
_output_shapes
:*
T0*
use_locking(*
validate_shape(*!
_class
loc:@v/dense_1/bias
�
save_9/Assign_59Assignv/dense_1/bias/Adamsave_9/RestoreV2:59*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:*
T0*
use_locking(
�
save_9/Assign_60Assignv/dense_1/bias/Adam_1save_9/RestoreV2:60*
use_locking(*
_output_shapes
:*
validate_shape(*!
_class
loc:@v/dense_1/bias*
T0
�
save_9/Assign_61Assignv/dense_1/kernelsave_9/RestoreV2:61*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
use_locking(*
validate_shape(*
T0
�
save_9/Assign_62Assignv/dense_1/kernel/Adamsave_9/RestoreV2:62*
validate_shape(*
use_locking(*#
_class
loc:@v/dense_1/kernel*
_output_shapes

: *
T0
�
save_9/Assign_63Assignv/dense_1/kernel/Adam_1save_9/RestoreV2:63*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
use_locking(
�
save_9/Assign_64Assignv/dense_2/biassave_9/RestoreV2:64*!
_class
loc:@v/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
�
save_9/Assign_65Assignv/dense_2/bias/Adamsave_9/RestoreV2:65*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias
�
save_9/Assign_66Assignv/dense_2/bias/Adam_1save_9/RestoreV2:66*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_9/Assign_67Assignv/dense_2/kernelsave_9/RestoreV2:67*
use_locking(*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
T0
�
save_9/Assign_68Assignv/dense_2/kernel/Adamsave_9/RestoreV2:68*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
use_locking(*
_output_shapes

:*
T0
�
save_9/Assign_69Assignv/dense_2/kernel/Adam_1save_9/RestoreV2:69*
T0*
use_locking(*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
validate_shape(
�
save_9/Assign_70Assignv/dense_3/biassave_9/RestoreV2:70*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
T0*
validate_shape(
�
save_9/Assign_71Assignv/dense_3/bias/Adamsave_9/RestoreV2:71*
_output_shapes
:*
T0*!
_class
loc:@v/dense_3/bias*
use_locking(*
validate_shape(
�
save_9/Assign_72Assignv/dense_3/bias/Adam_1save_9/RestoreV2:72*
use_locking(*
_output_shapes
:*!
_class
loc:@v/dense_3/bias*
T0*
validate_shape(
�
save_9/Assign_73Assignv/dense_3/kernelsave_9/RestoreV2:73*
T0*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
use_locking(*
validate_shape(
�
save_9/Assign_74Assignv/dense_3/kernel/Adamsave_9/RestoreV2:74*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
T0*
_output_shapes

:*
use_locking(
�
save_9/Assign_75Assignv/dense_3/kernel/Adam_1save_9/RestoreV2:75*
T0*
use_locking(*#
_class
loc:@v/dense_3/kernel*
validate_shape(*
_output_shapes

:
�
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_40^save_9/Assign_41^save_9/Assign_42^save_9/Assign_43^save_9/Assign_44^save_9/Assign_45^save_9/Assign_46^save_9/Assign_47^save_9/Assign_48^save_9/Assign_49^save_9/Assign_5^save_9/Assign_50^save_9/Assign_51^save_9/Assign_52^save_9/Assign_53^save_9/Assign_54^save_9/Assign_55^save_9/Assign_56^save_9/Assign_57^save_9/Assign_58^save_9/Assign_59^save_9/Assign_6^save_9/Assign_60^save_9/Assign_61^save_9/Assign_62^save_9/Assign_63^save_9/Assign_64^save_9/Assign_65^save_9/Assign_66^save_9/Assign_67^save_9/Assign_68^save_9/Assign_69^save_9/Assign_7^save_9/Assign_70^save_9/Assign_71^save_9/Assign_72^save_9/Assign_73^save_9/Assign_74^save_9/Assign_75^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
shape: *
_output_shapes
: *
dtype0
�
save_10/StringJoin/inputs_1Const*<
value3B1 B+_temp_51c24eb956d643ada827c78fbf2b310d/part*
_output_shapes
: *
dtype0
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_10/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_10/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
�
save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
�
save_10/SaveV2/tensor_namesConst*
_output_shapes
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0
�
save_10/SaveV2/shape_and_slicesConst*
dtype0*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:L
�
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi_j/dense/biaspi_j/dense/bias/Adampi_j/dense/bias/Adam_1pi_j/dense/kernelpi_j/dense/kernel/Adampi_j/dense/kernel/Adam_1pi_j/dense_1/biaspi_j/dense_1/bias/Adampi_j/dense_1/bias/Adam_1pi_j/dense_1/kernelpi_j/dense_1/kernel/Adampi_j/dense_1/kernel/Adam_1pi_j/dense_2/biaspi_j/dense_2/bias/Adampi_j/dense_2/bias/Adam_1pi_j/dense_2/kernelpi_j/dense_2/kernel/Adampi_j/dense_2/kernel/Adam_1pi_j/dense_3/biaspi_j/dense_3/bias/Adampi_j/dense_3/bias/Adam_1pi_j/dense_3/kernelpi_j/dense_3/kernel/Adampi_j/dense_3/kernel/Adam_1pi_n/dense/biaspi_n/dense/bias/Adampi_n/dense/bias/Adam_1pi_n/dense/kernelpi_n/dense/kernel/Adampi_n/dense/kernel/Adam_1pi_n/dense_1/biaspi_n/dense_1/bias/Adampi_n/dense_1/bias/Adam_1pi_n/dense_1/kernelpi_n/dense_1/kernel/Adampi_n/dense_1/kernel/Adam_1pi_n/dense_2/biaspi_n/dense_2/bias/Adampi_n/dense_2/bias/Adam_1pi_n/dense_2/kernelpi_n/dense_2/kernel/Adampi_n/dense_2/kernel/Adam_1pi_n/dense_3/biaspi_n/dense_3/bias/Adampi_n/dense_3/bias/Adam_1pi_n/dense_3/kernelpi_n/dense_3/kernel/Adampi_n/dense_3/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1v/dense_3/biasv/dense_3/bias/Adamv/dense_3/bias/Adam_1v/dense_3/kernelv/dense_3/kernel/Adamv/dense_3/kernel/Adam_1*Z
dtypesP
N2L
�
save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_10/ShardedFilename
�
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
N*

axis *
T0*
_output_shapes
:
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
:L*�
value�B�LBbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi_j/dense/biasBpi_j/dense/bias/AdamBpi_j/dense/bias/Adam_1Bpi_j/dense/kernelBpi_j/dense/kernel/AdamBpi_j/dense/kernel/Adam_1Bpi_j/dense_1/biasBpi_j/dense_1/bias/AdamBpi_j/dense_1/bias/Adam_1Bpi_j/dense_1/kernelBpi_j/dense_1/kernel/AdamBpi_j/dense_1/kernel/Adam_1Bpi_j/dense_2/biasBpi_j/dense_2/bias/AdamBpi_j/dense_2/bias/Adam_1Bpi_j/dense_2/kernelBpi_j/dense_2/kernel/AdamBpi_j/dense_2/kernel/Adam_1Bpi_j/dense_3/biasBpi_j/dense_3/bias/AdamBpi_j/dense_3/bias/Adam_1Bpi_j/dense_3/kernelBpi_j/dense_3/kernel/AdamBpi_j/dense_3/kernel/Adam_1Bpi_n/dense/biasBpi_n/dense/bias/AdamBpi_n/dense/bias/Adam_1Bpi_n/dense/kernelBpi_n/dense/kernel/AdamBpi_n/dense/kernel/Adam_1Bpi_n/dense_1/biasBpi_n/dense_1/bias/AdamBpi_n/dense_1/bias/Adam_1Bpi_n/dense_1/kernelBpi_n/dense_1/kernel/AdamBpi_n/dense_1/kernel/Adam_1Bpi_n/dense_2/biasBpi_n/dense_2/bias/AdamBpi_n/dense_2/bias/Adam_1Bpi_n/dense_2/kernelBpi_n/dense_2/kernel/AdamBpi_n/dense_2/kernel/Adam_1Bpi_n/dense_3/biasBpi_n/dense_3/bias/AdamBpi_n/dense_3/bias/Adam_1Bpi_n/dense_3/kernelBpi_n/dense_3/kernel/AdamBpi_n/dense_3/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1Bv/dense_3/biasBv/dense_3/bias/AdamBv/dense_3/bias/Adam_1Bv/dense_3/kernelBv/dense_3/kernel/AdamBv/dense_3/kernel/Adam_1*
dtype0
�
"save_10/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:L*�
value�B�LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*Z
dtypesP
N2L*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
�
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
validate_shape(*
_output_shapes
: *
T0*"
_class
loc:@pi_j/dense/bias*
use_locking(
�
save_10/Assign_1Assignbeta1_power_1save_10/RestoreV2:1*
validate_shape(*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: *
use_locking(
�
save_10/Assign_2Assignbeta2_powersave_10/RestoreV2:2*
_output_shapes
: *"
_class
loc:@pi_j/dense/bias*
validate_shape(*
T0*
use_locking(
�
save_10/Assign_3Assignbeta2_power_1save_10/RestoreV2:3*
use_locking(*
_class
loc:@v/dense/bias*
T0*
_output_shapes
: *
validate_shape(
�
save_10/Assign_4Assignpi_j/dense/biassave_10/RestoreV2:4*
validate_shape(*"
_class
loc:@pi_j/dense/bias*
use_locking(*
T0*
_output_shapes
: 
�
save_10/Assign_5Assignpi_j/dense/bias/Adamsave_10/RestoreV2:5*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi_j/dense/bias*
_output_shapes
: 
�
save_10/Assign_6Assignpi_j/dense/bias/Adam_1save_10/RestoreV2:6*"
_class
loc:@pi_j/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
�
save_10/Assign_7Assignpi_j/dense/kernelsave_10/RestoreV2:7*
use_locking(*
T0*$
_class
loc:@pi_j/dense/kernel*
_output_shapes

: *
validate_shape(
�
save_10/Assign_8Assignpi_j/dense/kernel/Adamsave_10/RestoreV2:8*
use_locking(*$
_class
loc:@pi_j/dense/kernel*
T0*
validate_shape(*
_output_shapes

: 
�
save_10/Assign_9Assignpi_j/dense/kernel/Adam_1save_10/RestoreV2:9*
validate_shape(*
use_locking(*$
_class
loc:@pi_j/dense/kernel*
T0*
_output_shapes

: 
�
save_10/Assign_10Assignpi_j/dense_1/biassave_10/RestoreV2:10*
_output_shapes
:*
validate_shape(*
T0*$
_class
loc:@pi_j/dense_1/bias*
use_locking(
�
save_10/Assign_11Assignpi_j/dense_1/bias/Adamsave_10/RestoreV2:11*
T0*$
_class
loc:@pi_j/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:
�
save_10/Assign_12Assignpi_j/dense_1/bias/Adam_1save_10/RestoreV2:12*
use_locking(*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_j/dense_1/bias*
T0
�
save_10/Assign_13Assignpi_j/dense_1/kernelsave_10/RestoreV2:13*
use_locking(*
T0*
_output_shapes

: *&
_class
loc:@pi_j/dense_1/kernel*
validate_shape(
�
save_10/Assign_14Assignpi_j/dense_1/kernel/Adamsave_10/RestoreV2:14*
validate_shape(*
use_locking(*
T0*&
_class
loc:@pi_j/dense_1/kernel*
_output_shapes

: 
�
save_10/Assign_15Assignpi_j/dense_1/kernel/Adam_1save_10/RestoreV2:15*&
_class
loc:@pi_j/dense_1/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes

: 
�
save_10/Assign_16Assignpi_j/dense_2/biassave_10/RestoreV2:16*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@pi_j/dense_2/bias*
validate_shape(
�
save_10/Assign_17Assignpi_j/dense_2/bias/Adamsave_10/RestoreV2:17*
use_locking(*
T0*
validate_shape(*
_output_shapes
:*$
_class
loc:@pi_j/dense_2/bias
�
save_10/Assign_18Assignpi_j/dense_2/bias/Adam_1save_10/RestoreV2:18*
T0*
_output_shapes
:*
validate_shape(*$
_class
loc:@pi_j/dense_2/bias*
use_locking(
�
save_10/Assign_19Assignpi_j/dense_2/kernelsave_10/RestoreV2:19*
use_locking(*
validate_shape(*
_output_shapes

:*&
_class
loc:@pi_j/dense_2/kernel*
T0
�
save_10/Assign_20Assignpi_j/dense_2/kernel/Adamsave_10/RestoreV2:20*
_output_shapes

:*
use_locking(*
T0*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel
�
save_10/Assign_21Assignpi_j/dense_2/kernel/Adam_1save_10/RestoreV2:21*
T0*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_j/dense_2/kernel*
use_locking(
�
save_10/Assign_22Assignpi_j/dense_3/biassave_10/RestoreV2:22*
validate_shape(*
T0*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_j/dense_3/bias
�
save_10/Assign_23Assignpi_j/dense_3/bias/Adamsave_10/RestoreV2:23*
validate_shape(*
use_locking(*
T0*$
_class
loc:@pi_j/dense_3/bias*
_output_shapes
:
�
save_10/Assign_24Assignpi_j/dense_3/bias/Adam_1save_10/RestoreV2:24*
_output_shapes
:*
use_locking(*
T0*$
_class
loc:@pi_j/dense_3/bias*
validate_shape(
�
save_10/Assign_25Assignpi_j/dense_3/kernelsave_10/RestoreV2:25*
_output_shapes

:*
use_locking(*&
_class
loc:@pi_j/dense_3/kernel*
T0*
validate_shape(
�
save_10/Assign_26Assignpi_j/dense_3/kernel/Adamsave_10/RestoreV2:26*
use_locking(*
T0*
validate_shape(*
_output_shapes

:*&
_class
loc:@pi_j/dense_3/kernel
�
save_10/Assign_27Assignpi_j/dense_3/kernel/Adam_1save_10/RestoreV2:27*
use_locking(*
_output_shapes

:*
validate_shape(*&
_class
loc:@pi_j/dense_3/kernel*
T0
�
save_10/Assign_28Assignpi_n/dense/biassave_10/RestoreV2:28*"
_class
loc:@pi_n/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
: *
T0
�
save_10/Assign_29Assignpi_n/dense/bias/Adamsave_10/RestoreV2:29*
validate_shape(*"
_class
loc:@pi_n/dense/bias*
use_locking(*
T0*
_output_shapes
: 
�
save_10/Assign_30Assignpi_n/dense/bias/Adam_1save_10/RestoreV2:30*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*"
_class
loc:@pi_n/dense/bias
�
save_10/Assign_31Assignpi_n/dense/kernelsave_10/RestoreV2:31*
use_locking(*
validate_shape(*
_output_shapes

: *$
_class
loc:@pi_n/dense/kernel*
T0
�
save_10/Assign_32Assignpi_n/dense/kernel/Adamsave_10/RestoreV2:32*$
_class
loc:@pi_n/dense/kernel*
validate_shape(*
use_locking(*
T0*
_output_shapes

: 
�
save_10/Assign_33Assignpi_n/dense/kernel/Adam_1save_10/RestoreV2:33*$
_class
loc:@pi_n/dense/kernel*
_output_shapes

: *
T0*
validate_shape(*
use_locking(
�
save_10/Assign_34Assignpi_n/dense_1/biassave_10/RestoreV2:34*
use_locking(*
T0*$
_class
loc:@pi_n/dense_1/bias*
_output_shapes
:*
validate_shape(
�
save_10/Assign_35Assignpi_n/dense_1/bias/Adamsave_10/RestoreV2:35*
validate_shape(*$
_class
loc:@pi_n/dense_1/bias*
use_locking(*
T0*
_output_shapes
:
�
save_10/Assign_36Assignpi_n/dense_1/bias/Adam_1save_10/RestoreV2:36*
_output_shapes
:*
T0*$
_class
loc:@pi_n/dense_1/bias*
use_locking(*
validate_shape(
�
save_10/Assign_37Assignpi_n/dense_1/kernelsave_10/RestoreV2:37*
validate_shape(*
T0*&
_class
loc:@pi_n/dense_1/kernel*
use_locking(*
_output_shapes

: 
�
save_10/Assign_38Assignpi_n/dense_1/kernel/Adamsave_10/RestoreV2:38*
_output_shapes

: *&
_class
loc:@pi_n/dense_1/kernel*
T0*
validate_shape(*
use_locking(
�
save_10/Assign_39Assignpi_n/dense_1/kernel/Adam_1save_10/RestoreV2:39*&
_class
loc:@pi_n/dense_1/kernel*
use_locking(*
T0*
_output_shapes

: *
validate_shape(
�
save_10/Assign_40Assignpi_n/dense_2/biassave_10/RestoreV2:40*
validate_shape(*
T0*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
_output_shapes
:
�
save_10/Assign_41Assignpi_n/dense_2/bias/Adamsave_10/RestoreV2:41*
validate_shape(*$
_class
loc:@pi_n/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
�
save_10/Assign_42Assignpi_n/dense_2/bias/Adam_1save_10/RestoreV2:42*
T0*
_output_shapes
:*$
_class
loc:@pi_n/dense_2/bias*
use_locking(*
validate_shape(
�
save_10/Assign_43Assignpi_n/dense_2/kernelsave_10/RestoreV2:43*
validate_shape(*&
_class
loc:@pi_n/dense_2/kernel*
use_locking(*
T0*
_output_shapes

:
�
save_10/Assign_44Assignpi_n/dense_2/kernel/Adamsave_10/RestoreV2:44*&
_class
loc:@pi_n/dense_2/kernel*
validate_shape(*
_output_shapes

:*
T0*
use_locking(
�
save_10/Assign_45Assignpi_n/dense_2/kernel/Adam_1save_10/RestoreV2:45*
use_locking(*&
_class
loc:@pi_n/dense_2/kernel*
T0*
_output_shapes

:*
validate_shape(
�
save_10/Assign_46Assignpi_n/dense_3/biassave_10/RestoreV2:46*$
_class
loc:@pi_n/dense_3/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0
�
save_10/Assign_47Assignpi_n/dense_3/bias/Adamsave_10/RestoreV2:47*
T0*
_output_shapes
:*
use_locking(*$
_class
loc:@pi_n/dense_3/bias*
validate_shape(
�
save_10/Assign_48Assignpi_n/dense_3/bias/Adam_1save_10/RestoreV2:48*
validate_shape(*$
_class
loc:@pi_n/dense_3/bias*
T0*
_output_shapes
:*
use_locking(
�
save_10/Assign_49Assignpi_n/dense_3/kernelsave_10/RestoreV2:49*
use_locking(*
validate_shape(*&
_class
loc:@pi_n/dense_3/kernel*
_output_shapes

:*
T0
�
save_10/Assign_50Assignpi_n/dense_3/kernel/Adamsave_10/RestoreV2:50*
_output_shapes

:*
use_locking(*
T0*
validate_shape(*&
_class
loc:@pi_n/dense_3/kernel
�
save_10/Assign_51Assignpi_n/dense_3/kernel/Adam_1save_10/RestoreV2:51*
validate_shape(*&
_class
loc:@pi_n/dense_3/kernel*
T0*
use_locking(*
_output_shapes

:
�
save_10/Assign_52Assignv/dense/biassave_10/RestoreV2:52*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0*
validate_shape(*
use_locking(
�
save_10/Assign_53Assignv/dense/bias/Adamsave_10/RestoreV2:53*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias*
validate_shape(*
use_locking(
�
save_10/Assign_54Assignv/dense/bias/Adam_1save_10/RestoreV2:54*
validate_shape(*
use_locking(*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
�
save_10/Assign_55Assignv/dense/kernelsave_10/RestoreV2:55*
validate_shape(*
_output_shapes
:	� *
use_locking(*!
_class
loc:@v/dense/kernel*
T0
�
save_10/Assign_56Assignv/dense/kernel/Adamsave_10/RestoreV2:56*!
_class
loc:@v/dense/kernel*
T0*
use_locking(*
_output_shapes
:	� *
validate_shape(
�
save_10/Assign_57Assignv/dense/kernel/Adam_1save_10/RestoreV2:57*
use_locking(*!
_class
loc:@v/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	� 
�
save_10/Assign_58Assignv/dense_1/biassave_10/RestoreV2:58*
T0*
_output_shapes
:*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(
�
save_10/Assign_59Assignv/dense_1/bias/Adamsave_10/RestoreV2:59*
validate_shape(*
use_locking(*
_output_shapes
:*
T0*!
_class
loc:@v/dense_1/bias
�
save_10/Assign_60Assignv/dense_1/bias/Adam_1save_10/RestoreV2:60*!
_class
loc:@v/dense_1/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
�
save_10/Assign_61Assignv/dense_1/kernelsave_10/RestoreV2:61*
_output_shapes

: *
use_locking(*
validate_shape(*
T0*#
_class
loc:@v/dense_1/kernel
�
save_10/Assign_62Assignv/dense_1/kernel/Adamsave_10/RestoreV2:62*
validate_shape(*
_output_shapes

: *#
_class
loc:@v/dense_1/kernel*
use_locking(*
T0
�
save_10/Assign_63Assignv/dense_1/kernel/Adam_1save_10/RestoreV2:63*
use_locking(*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
T0*
_output_shapes

: 
�
save_10/Assign_64Assignv/dense_2/biassave_10/RestoreV2:64*
validate_shape(*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(
�
save_10/Assign_65Assignv/dense_2/bias/Adamsave_10/RestoreV2:65*
T0*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
validate_shape(
�
save_10/Assign_66Assignv/dense_2/bias/Adam_1save_10/RestoreV2:66*
validate_shape(*
_output_shapes
:*!
_class
loc:@v/dense_2/bias*
use_locking(*
T0
�
save_10/Assign_67Assignv/dense_2/kernelsave_10/RestoreV2:67*
T0*
use_locking(*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel
�
save_10/Assign_68Assignv/dense_2/kernel/Adamsave_10/RestoreV2:68*
_output_shapes

:*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
T0*
use_locking(
�
save_10/Assign_69Assignv/dense_2/kernel/Adam_1save_10/RestoreV2:69*
T0*
use_locking(*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:*
validate_shape(
�
save_10/Assign_70Assignv/dense_3/biassave_10/RestoreV2:70*!
_class
loc:@v/dense_3/bias*
T0*
use_locking(*
_output_shapes
:*
validate_shape(
�
save_10/Assign_71Assignv/dense_3/bias/Adamsave_10/RestoreV2:71*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@v/dense_3/bias*
use_locking(
�
save_10/Assign_72Assignv/dense_3/bias/Adam_1save_10/RestoreV2:72*!
_class
loc:@v/dense_3/bias*
use_locking(*
validate_shape(*
_output_shapes
:*
T0
�
save_10/Assign_73Assignv/dense_3/kernelsave_10/RestoreV2:73*#
_class
loc:@v/dense_3/kernel*
_output_shapes

:*
T0*
validate_shape(*
use_locking(
�
save_10/Assign_74Assignv/dense_3/kernel/Adamsave_10/RestoreV2:74*
T0*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
use_locking(*
validate_shape(
�
save_10/Assign_75Assignv/dense_3/kernel/Adam_1save_10/RestoreV2:75*
use_locking(*
validate_shape(*
_output_shapes

:*#
_class
loc:@v/dense_3/kernel*
T0
�
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_40^save_10/Assign_41^save_10/Assign_42^save_10/Assign_43^save_10/Assign_44^save_10/Assign_45^save_10/Assign_46^save_10/Assign_47^save_10/Assign_48^save_10/Assign_49^save_10/Assign_5^save_10/Assign_50^save_10/Assign_51^save_10/Assign_52^save_10/Assign_53^save_10/Assign_54^save_10/Assign_55^save_10/Assign_56^save_10/Assign_57^save_10/Assign_58^save_10/Assign_59^save_10/Assign_6^save_10/Assign_60^save_10/Assign_61^save_10/Assign_62^save_10/Assign_63^save_10/Assign_64^save_10/Assign_65^save_10/Assign_66^save_10/Assign_67^save_10/Assign_68^save_10/Assign_69^save_10/Assign_7^save_10/Assign_70^save_10/Assign_71^save_10/Assign_72^save_10/Assign_73^save_10/Assign_74^save_10/Assign_75^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard "&E
save_10/Const:0save_10/Identity:0save_10/restore_all (5 @F8"�
trainable_variables��
{
pi_j/dense/kernel:0pi_j/dense/kernel/Assignpi_j/dense/kernel/read:02.pi_j/dense/kernel/Initializer/random_uniform:08
j
pi_j/dense/bias:0pi_j/dense/bias/Assignpi_j/dense/bias/read:02#pi_j/dense/bias/Initializer/zeros:08
�
pi_j/dense_1/kernel:0pi_j/dense_1/kernel/Assignpi_j/dense_1/kernel/read:020pi_j/dense_1/kernel/Initializer/random_uniform:08
r
pi_j/dense_1/bias:0pi_j/dense_1/bias/Assignpi_j/dense_1/bias/read:02%pi_j/dense_1/bias/Initializer/zeros:08
�
pi_j/dense_2/kernel:0pi_j/dense_2/kernel/Assignpi_j/dense_2/kernel/read:020pi_j/dense_2/kernel/Initializer/random_uniform:08
r
pi_j/dense_2/bias:0pi_j/dense_2/bias/Assignpi_j/dense_2/bias/read:02%pi_j/dense_2/bias/Initializer/zeros:08
�
pi_j/dense_3/kernel:0pi_j/dense_3/kernel/Assignpi_j/dense_3/kernel/read:020pi_j/dense_3/kernel/Initializer/random_uniform:08
r
pi_j/dense_3/bias:0pi_j/dense_3/bias/Assignpi_j/dense_3/bias/read:02%pi_j/dense_3/bias/Initializer/zeros:08
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
{
pi_n/dense/kernel:0pi_n/dense/kernel/Assignpi_n/dense/kernel/read:02.pi_n/dense/kernel/Initializer/random_uniform:08
j
pi_n/dense/bias:0pi_n/dense/bias/Assignpi_n/dense/bias/read:02#pi_n/dense/bias/Initializer/zeros:08
�
pi_n/dense_1/kernel:0pi_n/dense_1/kernel/Assignpi_n/dense_1/kernel/read:020pi_n/dense_1/kernel/Initializer/random_uniform:08
r
pi_n/dense_1/bias:0pi_n/dense_1/bias/Assignpi_n/dense_1/bias/read:02%pi_n/dense_1/bias/Initializer/zeros:08
�
pi_n/dense_2/kernel:0pi_n/dense_2/kernel/Assignpi_n/dense_2/kernel/read:020pi_n/dense_2/kernel/Initializer/random_uniform:08
r
pi_n/dense_2/bias:0pi_n/dense_2/bias/Assignpi_n/dense_2/bias/read:02%pi_n/dense_2/bias/Initializer/zeros:08
�
pi_n/dense_3/kernel:0pi_n/dense_3/kernel/Assignpi_n/dense_3/kernel/read:020pi_n/dense_3/kernel/Initializer/random_uniform:08
r
pi_n/dense_3/bias:0pi_n/dense_3/bias/Assignpi_n/dense_3/bias/read:02%pi_n/dense_3/bias/Initializer/zeros:08"
train_v


Adam_1"
train_pi

Adam"�L
	variables�L�L
{
pi_j/dense/kernel:0pi_j/dense/kernel/Assignpi_j/dense/kernel/read:02.pi_j/dense/kernel/Initializer/random_uniform:08
j
pi_j/dense/bias:0pi_j/dense/bias/Assignpi_j/dense/bias/read:02#pi_j/dense/bias/Initializer/zeros:08
�
pi_j/dense_1/kernel:0pi_j/dense_1/kernel/Assignpi_j/dense_1/kernel/read:020pi_j/dense_1/kernel/Initializer/random_uniform:08
r
pi_j/dense_1/bias:0pi_j/dense_1/bias/Assignpi_j/dense_1/bias/read:02%pi_j/dense_1/bias/Initializer/zeros:08
�
pi_j/dense_2/kernel:0pi_j/dense_2/kernel/Assignpi_j/dense_2/kernel/read:020pi_j/dense_2/kernel/Initializer/random_uniform:08
r
pi_j/dense_2/bias:0pi_j/dense_2/bias/Assignpi_j/dense_2/bias/read:02%pi_j/dense_2/bias/Initializer/zeros:08
�
pi_j/dense_3/kernel:0pi_j/dense_3/kernel/Assignpi_j/dense_3/kernel/read:020pi_j/dense_3/kernel/Initializer/random_uniform:08
r
pi_j/dense_3/bias:0pi_j/dense_3/bias/Assignpi_j/dense_3/bias/read:02%pi_j/dense_3/bias/Initializer/zeros:08
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
{
pi_n/dense/kernel:0pi_n/dense/kernel/Assignpi_n/dense/kernel/read:02.pi_n/dense/kernel/Initializer/random_uniform:08
j
pi_n/dense/bias:0pi_n/dense/bias/Assignpi_n/dense/bias/read:02#pi_n/dense/bias/Initializer/zeros:08
�
pi_n/dense_1/kernel:0pi_n/dense_1/kernel/Assignpi_n/dense_1/kernel/read:020pi_n/dense_1/kernel/Initializer/random_uniform:08
r
pi_n/dense_1/bias:0pi_n/dense_1/bias/Assignpi_n/dense_1/bias/read:02%pi_n/dense_1/bias/Initializer/zeros:08
�
pi_n/dense_2/kernel:0pi_n/dense_2/kernel/Assignpi_n/dense_2/kernel/read:020pi_n/dense_2/kernel/Initializer/random_uniform:08
r
pi_n/dense_2/bias:0pi_n/dense_2/bias/Assignpi_n/dense_2/bias/read:02%pi_n/dense_2/bias/Initializer/zeros:08
�
pi_n/dense_3/kernel:0pi_n/dense_3/kernel/Assignpi_n/dense_3/kernel/read:020pi_n/dense_3/kernel/Initializer/random_uniform:08
r
pi_n/dense_3/bias:0pi_n/dense_3/bias/Assignpi_n/dense_3/bias/read:02%pi_n/dense_3/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
�
pi_j/dense/kernel/Adam:0pi_j/dense/kernel/Adam/Assignpi_j/dense/kernel/Adam/read:02*pi_j/dense/kernel/Adam/Initializer/zeros:0
�
pi_j/dense/kernel/Adam_1:0pi_j/dense/kernel/Adam_1/Assignpi_j/dense/kernel/Adam_1/read:02,pi_j/dense/kernel/Adam_1/Initializer/zeros:0
|
pi_j/dense/bias/Adam:0pi_j/dense/bias/Adam/Assignpi_j/dense/bias/Adam/read:02(pi_j/dense/bias/Adam/Initializer/zeros:0
�
pi_j/dense/bias/Adam_1:0pi_j/dense/bias/Adam_1/Assignpi_j/dense/bias/Adam_1/read:02*pi_j/dense/bias/Adam_1/Initializer/zeros:0
�
pi_j/dense_1/kernel/Adam:0pi_j/dense_1/kernel/Adam/Assignpi_j/dense_1/kernel/Adam/read:02,pi_j/dense_1/kernel/Adam/Initializer/zeros:0
�
pi_j/dense_1/kernel/Adam_1:0!pi_j/dense_1/kernel/Adam_1/Assign!pi_j/dense_1/kernel/Adam_1/read:02.pi_j/dense_1/kernel/Adam_1/Initializer/zeros:0
�
pi_j/dense_1/bias/Adam:0pi_j/dense_1/bias/Adam/Assignpi_j/dense_1/bias/Adam/read:02*pi_j/dense_1/bias/Adam/Initializer/zeros:0
�
pi_j/dense_1/bias/Adam_1:0pi_j/dense_1/bias/Adam_1/Assignpi_j/dense_1/bias/Adam_1/read:02,pi_j/dense_1/bias/Adam_1/Initializer/zeros:0
�
pi_j/dense_2/kernel/Adam:0pi_j/dense_2/kernel/Adam/Assignpi_j/dense_2/kernel/Adam/read:02,pi_j/dense_2/kernel/Adam/Initializer/zeros:0
�
pi_j/dense_2/kernel/Adam_1:0!pi_j/dense_2/kernel/Adam_1/Assign!pi_j/dense_2/kernel/Adam_1/read:02.pi_j/dense_2/kernel/Adam_1/Initializer/zeros:0
�
pi_j/dense_2/bias/Adam:0pi_j/dense_2/bias/Adam/Assignpi_j/dense_2/bias/Adam/read:02*pi_j/dense_2/bias/Adam/Initializer/zeros:0
�
pi_j/dense_2/bias/Adam_1:0pi_j/dense_2/bias/Adam_1/Assignpi_j/dense_2/bias/Adam_1/read:02,pi_j/dense_2/bias/Adam_1/Initializer/zeros:0
�
pi_j/dense_3/kernel/Adam:0pi_j/dense_3/kernel/Adam/Assignpi_j/dense_3/kernel/Adam/read:02,pi_j/dense_3/kernel/Adam/Initializer/zeros:0
�
pi_j/dense_3/kernel/Adam_1:0!pi_j/dense_3/kernel/Adam_1/Assign!pi_j/dense_3/kernel/Adam_1/read:02.pi_j/dense_3/kernel/Adam_1/Initializer/zeros:0
�
pi_j/dense_3/bias/Adam:0pi_j/dense_3/bias/Adam/Assignpi_j/dense_3/bias/Adam/read:02*pi_j/dense_3/bias/Adam/Initializer/zeros:0
�
pi_j/dense_3/bias/Adam_1:0pi_j/dense_3/bias/Adam_1/Assignpi_j/dense_3/bias/Adam_1/read:02,pi_j/dense_3/bias/Adam_1/Initializer/zeros:0
�
pi_n/dense/kernel/Adam:0pi_n/dense/kernel/Adam/Assignpi_n/dense/kernel/Adam/read:02*pi_n/dense/kernel/Adam/Initializer/zeros:0
�
pi_n/dense/kernel/Adam_1:0pi_n/dense/kernel/Adam_1/Assignpi_n/dense/kernel/Adam_1/read:02,pi_n/dense/kernel/Adam_1/Initializer/zeros:0
|
pi_n/dense/bias/Adam:0pi_n/dense/bias/Adam/Assignpi_n/dense/bias/Adam/read:02(pi_n/dense/bias/Adam/Initializer/zeros:0
�
pi_n/dense/bias/Adam_1:0pi_n/dense/bias/Adam_1/Assignpi_n/dense/bias/Adam_1/read:02*pi_n/dense/bias/Adam_1/Initializer/zeros:0
�
pi_n/dense_1/kernel/Adam:0pi_n/dense_1/kernel/Adam/Assignpi_n/dense_1/kernel/Adam/read:02,pi_n/dense_1/kernel/Adam/Initializer/zeros:0
�
pi_n/dense_1/kernel/Adam_1:0!pi_n/dense_1/kernel/Adam_1/Assign!pi_n/dense_1/kernel/Adam_1/read:02.pi_n/dense_1/kernel/Adam_1/Initializer/zeros:0
�
pi_n/dense_1/bias/Adam:0pi_n/dense_1/bias/Adam/Assignpi_n/dense_1/bias/Adam/read:02*pi_n/dense_1/bias/Adam/Initializer/zeros:0
�
pi_n/dense_1/bias/Adam_1:0pi_n/dense_1/bias/Adam_1/Assignpi_n/dense_1/bias/Adam_1/read:02,pi_n/dense_1/bias/Adam_1/Initializer/zeros:0
�
pi_n/dense_2/kernel/Adam:0pi_n/dense_2/kernel/Adam/Assignpi_n/dense_2/kernel/Adam/read:02,pi_n/dense_2/kernel/Adam/Initializer/zeros:0
�
pi_n/dense_2/kernel/Adam_1:0!pi_n/dense_2/kernel/Adam_1/Assign!pi_n/dense_2/kernel/Adam_1/read:02.pi_n/dense_2/kernel/Adam_1/Initializer/zeros:0
�
pi_n/dense_2/bias/Adam:0pi_n/dense_2/bias/Adam/Assignpi_n/dense_2/bias/Adam/read:02*pi_n/dense_2/bias/Adam/Initializer/zeros:0
�
pi_n/dense_2/bias/Adam_1:0pi_n/dense_2/bias/Adam_1/Assignpi_n/dense_2/bias/Adam_1/read:02,pi_n/dense_2/bias/Adam_1/Initializer/zeros:0
�
pi_n/dense_3/kernel/Adam:0pi_n/dense_3/kernel/Adam/Assignpi_n/dense_3/kernel/Adam/read:02,pi_n/dense_3/kernel/Adam/Initializer/zeros:0
�
pi_n/dense_3/kernel/Adam_1:0!pi_n/dense_3/kernel/Adam_1/Assign!pi_n/dense_3/kernel/Adam_1/read:02.pi_n/dense_3/kernel/Adam_1/Initializer/zeros:0
�
pi_n/dense_3/bias/Adam:0pi_n/dense_3/bias/Adam/Assignpi_n/dense_3/bias/Adam/read:02*pi_n/dense_3/bias/Adam/Initializer/zeros:0
�
pi_n/dense_3/bias/Adam_1:0pi_n/dense_3/bias/Adam_1/Assignpi_n/dense_3/bias/Adam_1/read:02,pi_n/dense_3/bias/Adam_1/Initializer/zeros:0
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
v/dense_3/bias/Adam_1:0v/dense_3/bias/Adam_1/Assignv/dense_3/bias/Adam_1/read:02)v/dense_3/bias/Adam_1/Initializer/zeros:0"
train_op

Adam
Adam_1*�	
serving_default�	
3
logpn_old_ph#
Placeholder_10:0���������
2
logpj_old_ph"
Placeholder_9:0���������
)
adv"
Placeholder_7:0���������
(
an"
Placeholder_4:0���������
-
xn'
Placeholder_1:0����������
0
maskj'
Placeholder_5:0����������
-
co'
Placeholder_2:0����������
+
xj%
Placeholder:0����������
/
maskn&
Placeholder_6:0���������
)
ret"
Placeholder_8:0���������
(
aj"
Placeholder_3:0���������&
logpj

pi_j/Sum:0���������
pi_loss
Neg:0 +
pi_j#
pi_j/Squeeze_1:0	���������

clipfrac_n
Mean_7:0 -
	clipped_n 
LogicalOr_1:0
���������
approx_ent_j
Mean_3:0 
v_loss
Mean_1:0 +
pi_n#
pi_n/Squeeze_1:0	���������
approx_kl_j
Mean_2:0 +
	clipped_j
LogicalOr:0
���������+
out_j"

pi_j/add:0����������
approx_ent_n
Mean_6:0 ,
	logp_pi_n
pi_n/Sum_1:0���������#
v
v/Squeeze:0���������*
out_n!

pi_n/add:0���������

clipfrac_j
Mean_4:0 &
logpn

pi_n/Sum:0���������
approx_kl_n
Mean_5:0 ,
	logp_pi_j
pi_j/Sum_1:0���������tensorflow/serving/predict