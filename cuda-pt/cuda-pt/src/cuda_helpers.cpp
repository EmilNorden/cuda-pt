#include "error_assertion.h"
#include "cuda.h"
#include "driver_types.h"
#include <SDL.h>
#include <sstream>
#include <SDL_opengl.h>

void reportErrorAndDie(cudaError_t error, const char *file, int line)
{
	std::stringstream ss;
	ss << file << ":" << line << " - CUDA call returned error: " << cudaErrorToString(error);
	SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "CUDA PT Error", ss.str().c_str(), NULL);
	exit(EXIT_FAILURE);
}

void reportCudaDriverErrorAndDie(CUresult error, const char *file, int line)
{
	std::stringstream ss;
	ss << file << ":" << line << " - CUDA Driver call returned error: " << error;
	SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "CUDA PT Error", ss.str().c_str(), NULL);
	exit(EXIT_FAILURE);
}

void reportGLErrorAndDie(GLenum error, const char *file, int line)
{
	std::stringstream ss;
	ss << file << ":" << line << " - OpenGL call returned error: " << error;
	SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "CUDA PT Error", ss.str().c_str(), NULL);
	exit(EXIT_FAILURE);
}

const char *cudaErrorToString(cudaError_t error)
{
	switch(error){
	case cudaSuccess:
		return "cudaSuccess";
		break;
	case cudaErrorMissingConfiguration:
		return "cudaErrorMissingConfiguration";
		break;
	case cudaErrorMemoryAllocation:
		return "cudaErrorMemoryAllocation";
		break;
	case cudaErrorInitializationError:
		return "cudaErrorInitializationError";
		break;
	case cudaErrorLaunchFailure:
		return "cudaErrorLaunchFailure";
		break;
	case cudaErrorPriorLaunchFailure:
		return "cudaErrorPriorLaunchFailure";
		break;
	case cudaErrorLaunchTimeout:
		return "cudaErrorLaunchTimeout";
		break;
	case cudaErrorLaunchOutOfResources:
		return "cudaErrorLaunchOutOfResources";
		break;
	case cudaErrorInvalidDeviceFunction:
		return "cudaErrorInvalidDeviceFunction";
		break;
	case cudaErrorInvalidConfiguration:
		return "cudaErrorInvalidConfiguration";
		break;
	case cudaErrorInvalidDevice:
		return "cudaErrorInvalidDevice";
		break;
	case cudaErrorInvalidValue:
		return "cudaErrorInvalidValue";
		break;
	case cudaErrorInvalidPitchValue:
		return "cudaErrorInvalidPitchValue";
		break;
	case cudaErrorInvalidSymbol:
		return "cudaErrorInvalidSymbol";
		break;
	case cudaErrorMapBufferObjectFailed:
		return "cudaErrorMapBufferObjectFailed";
		break;
	case cudaErrorUnmapBufferObjectFailed:
		return "cudaErrorUnmapBufferObjectFailed";
		break;
	case cudaErrorInvalidHostPointer:
		return "cudaErrorInvalidHostPointer";
		break;
	case cudaErrorInvalidDevicePointer:
		return "cudaErrorInvalidDevicePointer";
		break;
	case cudaErrorInvalidTexture:
		return "cudaErrorInvalidTexture";
		break;
	case cudaErrorInvalidTextureBinding:
		return "cudaErrorInvalidTextureBinding";
		break;
	case cudaErrorInvalidChannelDescriptor:
		return "cudaErrorInvalidChannelDescriptor";
		break;
	case cudaErrorInvalidMemcpyDirection:
		return "cudaErrorInvalidMemcpyDirection";
		break;
	case cudaErrorAddressOfConstant:
		return "cudaErrorAddressOfConstant";
		break;
	case cudaErrorTextureFetchFailed:
		return "cudaErrorTextureFetchFailed";
		break;
	case cudaErrorTextureNotBound:
		return "cudaErrorTextureNotBound";
		break;
	case cudaErrorSynchronizationError:
		return "cudaErrorSynchronizationError";
		break;
	case cudaErrorInvalidFilterSetting:
		return "cudaErrorInvalidFilterSetting";
		break;
	case cudaErrorInvalidNormSetting:
		return "cudaErrorInvalidNormSetting";
		break;
	case cudaErrorMixedDeviceExecution:
		return "cudaErrorMixedDeviceExecution";
		break;
	case cudaErrorCudartUnloading:
		return "cudaErrorCudartUnloading";
		break;
	case cudaErrorUnknown:
		return "cudaErrorUnknown";
		break;
	case cudaErrorNotYetImplemented:
		return "cudaErrorNotYetImplemented";
		break;
	case cudaErrorMemoryValueTooLarge:
		return "cudaErrorMemoryValueTooLarge";
		break;
	case cudaErrorInvalidResourceHandle:
		return "cudaErrorInvalidResourceHandle";
		break;
	case cudaErrorNotReady:
		return "cudaErrorNotReady";
		break;
	case cudaErrorInsufficientDriver:
		return "cudaErrorInsufficientDriver";
		break;
	case cudaErrorSetOnActiveProcess:
		return "cudaErrorSetOnActiveProcess";
		break;
	case cudaErrorInvalidSurface:
		return "cudaErrorInvalidSurface";
		break;
	case cudaErrorNoDevice:
		return "cudaErrorNoDevice";
		break;
	case cudaErrorECCUncorrectable:
		return "cudaErrorECCUncorrectable";
		break;
	case cudaErrorSharedObjectSymbolNotFound:
		return "cudaErrorSharedObjectSymbolNotFound";
		break;
	case cudaErrorSharedObjectInitFailed:
		return "cudaErrorSharedObjectInitFailed";
		break;
	case cudaErrorUnsupportedLimit:
		return "cudaErrorUnsupportedLimit";
		break;
	case cudaErrorDuplicateVariableName:
		return "cudaErrorDuplicateVariableName";
		break;
	case cudaErrorDuplicateTextureName:
		return "cudaErrorDuplicateTextureName";
		break;
	case cudaErrorDuplicateSurfaceName:
		return "cudaErrorDuplicateSurfaceName";
		break;
	case cudaErrorDevicesUnavailable:
		return "cudaErrorDevicesUnavailable";
		break;
	case cudaErrorInvalidKernelImage:
		return "cudaErrorInvalidKernelImage";
		break;
	case cudaErrorNoKernelImageForDevice:
		return "cudaErrorNoKernelImageForDevice";
		break;
	case cudaErrorIncompatibleDriverContext:
		return "cudaErrorIncompatibleDriverContext";
		break;
	case cudaErrorPeerAccessAlreadyEnabled:
		return "cudaErrorPeerAccessAlreadyEnabled";
		break;
	case cudaErrorPeerAccessNotEnabled:
		return "cudaErrorPeerAccessNotEnabled";
		break;
	case cudaErrorDeviceAlreadyInUse:
		return "cudaErrorDeviceAlreadyInUse";
		break;
	case cudaErrorProfilerDisabled:
		return "cudaErrorProfilerDisabled";
		break;
	case cudaErrorProfilerNotInitialized:
		return "cudaErrorProfilerNotInitialized";
		break;
	case cudaErrorProfilerAlreadyStarted:
		return "cudaErrorProfilerAlreadyStarted";
		break;
	case cudaErrorProfilerAlreadyStopped:
		return "cudaErrorProfilerAlreadyStopped";
		break;
	case cudaErrorAssert:
		return "cudaErrorAssert";
		break;
	case cudaErrorTooManyPeers:
		return "cudaErrorTooManyPeers";
		break;
	case cudaErrorHostMemoryAlreadyRegistered:
		return "cudaErrorHostMemoryAlreadyRegistered";
		break;
	case cudaErrorHostMemoryNotRegistered:
		return "cudaErrorHostMemoryNotRegistered";
		break;
	case cudaErrorOperatingSystem:
		return "cudaErrorOperatingSystem";
		break;
	case cudaErrorPeerAccessUnsupported:
		return "cudaErrorPeerAccessUnsupported";
		break;
	case cudaErrorLaunchMaxDepthExceeded:
		return "cudaErrorLaunchMaxDepthExceeded";
		break;
	case cudaErrorLaunchFileScopedTex:
		return "cudaErrorLaunchFileScopedTex";
		break;
	case cudaErrorLaunchFileScopedSurf:
		return "cudaErrorLaunchFileScopedSurf";
		break;
	case cudaErrorSyncDepthExceeded:
		return "cudaErrorSyncDepthExceeded";
		break;
	case cudaErrorLaunchPendingCountExceeded:
		return "cudaErrorLaunchPendingCountExceeded";
		break;
	case cudaErrorNotPermitted:
		return "cudaErrorNotPermitted";
		break;
	case cudaErrorNotSupported:
		return "cudaErrorNotSupported";
		break;
	case cudaErrorStartupFailure:
		return "cudaErrorStartupFailure";
		break;
	case cudaErrorApiFailureBase:
		return "cudaErrorApiFailureBase";
		break;
	}
}