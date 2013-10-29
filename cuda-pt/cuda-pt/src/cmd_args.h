#ifndef CMD_ARGS_H_
#define CMD_ARGS_H_

class CmdArgs
{
public:
	int cuda_device;
	CmdArgs(int argc, char **argv);
};

#endif