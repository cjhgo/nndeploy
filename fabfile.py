import os
from fabric.context_managers import lcd, cd
from fabric.operations import local
from fabric.api import run, env, hosts
env.use_ssh_config=True
# env.hosts = ["m1sd"]


from pathlib import Path
repopath = os.getcwd()
stem = os.path.basename(repopath)
home = os.getenv('HOME')
rel_home = os.path.relpath(repopath, os.path.expanduser('~'))
print(home, repopath, stem)

build_ssh="m1sd"
run_ssh="m1"

# sync_dir="/tmp/cppinferlib"
sync_dir=f"/tmp/{stem}"
rsync_cmd="""rsync -avzk --exclude '.git'  --exclude build  """

def s2pdev():
    cmd=f"{rsync_cmd} ./ {build_ssh}:{sync_dir}"
    local(cmd)

def s2m1():
    cmd=f"rsync -avzk {build_ssh}:{sync_dir}/build/install/lib /tmp"
    local(cmd)
    cmd=f"rsync -av /tmp/lib {run_ssh}:/tmp"
    local(cmd)

def lbuild():
    """build  on local"""
    local('cmake -B build -C cmake/config_mac.cmake .')
    local('make -j -C build VERBOSE=TRUE')


@hosts(build_ssh)
def m1build():
    """build  on pdev"""
    s2pdev()
    with cd(sync_dir):
        cmake_cmd=f"cmake -B build -C cmake/config_rkm1.cmake ."
        run(cmake_cmd)
        run('make -j -C build  VERBOSE=TRUE')
        run('make install -j -C build  VERBOSE=TRUE')
    s2m1()

@hosts(build_ssh)
def babuild():
    """build  on pdev"""
    s2pdev()
    with cd(sync_dir):
        cmake_cmd=f"cmake -B build -C cmake/config_rkba.cmake ."
        run(cmake_cmd)
        run('make -j -C build  VERBOSE=TRUE')
        run('make install -j -C build  VERBOSE=TRUE')
    s2m1()


@hosts(build_ssh)
def akbuild():
    """build  on pdev"""
    s2pdev()
    with cd(sync_dir):
        cmake_cmd=f"cmake -B build -C cmake/config_rkm1.cmake ."
        run(cmake_cmd)
        run('make -j -C build  VERBOSE=TRUE')
        run('make install -j -C build  VERBOSE=TRUE')
    s2m1()

@hosts(run_ssh)
def runexe():
    with cd('/tmp/lib'):
        cmd=f"""./clfg --name clfg --inference_type kInferenceTypeRknn \
--device_type kDeviceTypeCodeCpu:0 --model_type kModelTypeRknn --is_path \
--model_value  /tmp/lib/out.rknn \
--codec_flag kCodecFlagImage --parallel_type kParallelTypeSequential \
--input_path /tmp/lib/in.png \
--output_path /tmp/out.png """
        run(cmd)
