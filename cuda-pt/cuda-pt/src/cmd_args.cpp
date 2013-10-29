#include "cmd_args.h"
#include <string>

CmdArgs::CmdArgs(int argc, char **argv)
{
	for(int i = 0; i < argc; ++i)
	{
		std::string argument(argv[i]);
		auto index = argument.find('=');
		if(index == std::string::npos)
		{
		}
	}
}