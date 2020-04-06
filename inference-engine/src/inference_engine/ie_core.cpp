// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core.hpp"

#include <unordered_set>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include <ngraph/opsets/opset.hpp>
#include "cpp_interfaces/base/ie_plugin_base.hpp"
#include "details/caseless.hpp"
#include "details/ie_exception_conversion.hpp"
#include "details/ie_so_pointer.hpp"
#include "file_utils.h"
#include "ie_cnn_net_reader_impl.h"
#include "ie_icore.hpp"
#include "ie_ir_reader.hpp"
#include "ie_plugin.hpp"
#include "ie_plugin_config.hpp"
#include "ie_profiling.hpp"
#include "ie_util_internal.hpp"
#include "multi-device/multi_device_config.hpp"
#include "xml_parse_utils.h"

using namespace InferenceEngine::PluginConfigParams;

namespace InferenceEngine {

IE_SUPPRESS_DEPRECATED_START

namespace {

IInferencePluginAPI* getInferencePluginAPIInterface(IInferencePlugin* iplugin) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  IInferencePluginAPI* getInferencePluginAPIInterface(IInferencePlugin* iplugin) {" << std::endl;
    return dynamic_cast<IInferencePluginAPI*>(iplugin);
}

IInferencePluginAPI* getInferencePluginAPIInterface(InferenceEnginePluginPtr iplugin) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  IInferencePluginAPI* getInferencePluginAPIInterface(InferenceEnginePluginPtr iplugin) {" << std::endl;
    return getInferencePluginAPIInterface(static_cast<IInferencePlugin*>(iplugin.operator->()));
}

IInferencePluginAPI* getInferencePluginAPIInterface(InferencePlugin plugin) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  IInferencePluginAPI* getInferencePluginAPIInterface(InferencePlugin plugin) {" << std::endl;
    return getInferencePluginAPIInterface(static_cast<InferenceEnginePluginPtr>(plugin));
}

}  // namespace

IInferencePlugin::~IInferencePlugin() {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  IInferencePlugin::~IInferencePlugin() {" << std::endl;}

IInferencePluginAPI::~IInferencePluginAPI() {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  IInferencePluginAPI::~IInferencePluginAPI() {" << std::endl;}

IE_SUPPRESS_DEPRECATED_END

DeviceIDParser::DeviceIDParser(const std::string& deviceNameWithID) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  DeviceIDParser::DeviceIDParser(const std::string& deviceNameWithID) {" << std::endl;
    deviceName = deviceNameWithID;

    auto pos = deviceName.find('.');
    if (pos != std::string::npos) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (pos != std::string::npos) {" << std::endl;
        deviceName = deviceNameWithID.substr(0, pos);
        deviceID = deviceNameWithID.substr(pos + 1, deviceNameWithID.size());
    }
}

std::string DeviceIDParser::getDeviceID() const {
    return deviceID;
}

std::string DeviceIDParser::getDeviceName() const {
    return deviceName;
}

std::vector<std::string> DeviceIDParser::getHeteroDevices(std::string fallbackDevice) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  std::vector<std::string> DeviceIDParser::getHeteroDevices(std::string fallbackDevice) {" << std::endl;
    std::vector<std::string> deviceNames;

    std::string cdevice;
    char delimiter = ',';
    size_t pos = 0;

    while ((pos = fallbackDevice.find(delimiter)) != std::string::npos) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      while ((pos = fallbackDevice.find(delimiter)) != std::string::npos) {" << std::endl;
        deviceNames.push_back(fallbackDevice.substr(0, pos));
        fallbackDevice.erase(0, pos + 1);
    }

    if (!fallbackDevice.empty()) deviceNames.push_back(fallbackDevice);

    return deviceNames;
}

std::vector<std::string> DeviceIDParser::getMultiDevices(std::string devicesList) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  std::vector<std::string> DeviceIDParser::getMultiDevices(std::string devicesList) {" << std::endl;
    std::vector<std::string> deviceNames;
    auto trim_request_info = [](std::string device_with_requests) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      auto trim_request_info = [](std::string device_with_requests) {" << std::endl;
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    std::string device;
    char delimiter = ',';
    size_t pos = 0;
    // in addition to the list of devices, every device can have a #requests in the brackets e.g. "CPU(100)"
    // we skip the #requests info here
    while ((pos = devicesList.find(delimiter)) != std::string::npos) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      while ((pos = devicesList.find(delimiter)) != std::string::npos) {" << std::endl;
        auto d = devicesList.substr(0, pos);
        deviceNames.push_back(trim_request_info(d));
        devicesList.erase(0, pos + 1);
    }

    if (!devicesList.empty()) deviceNames.push_back(trim_request_info(devicesList));

    return deviceNames;
}

class Core::Impl : public ICore {
    ITaskExecutor::Ptr _taskExecutor = nullptr;

    IE_SUPPRESS_DEPRECATED_START
    mutable std::map<std::string, InferencePlugin, details::CaselessLess<std::string>> plugins;
    IE_SUPPRESS_DEPRECATED_END

    struct PluginDescriptor {
        FileUtils::FilePath libraryLocation;
        std::map<std::string, std::string> defaultConfig;
        std::vector<FileUtils::FilePath> listOfExtentions;
    };

    std::map<std::string, PluginDescriptor, details::CaselessLess<std::string>> pluginRegistry;
    IErrorListener* listener = nullptr;
    std::unordered_set<std::string> opsetNames;
    std::vector<IExtensionPtr> extensions;

public:
    Impl();
    ~Impl() override;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file. The function supports UNICODE path
     * @param xmlConfigFile - an .xml configuraion with device / plugin information
     */
    void RegisterPluginsInRegistry(const std::string& xmlConfigFile) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      void RegisterPluginsInRegistry(const std::string& xmlConfigFile) {" << std::endl;
        std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry: begin" << std::endl;
        auto parse_result = ParseXml(xmlConfigFile.c_str());
        if (!parse_result.error_msg.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (!parse_result.error_msg.empty()) {" << std::endl;
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry: THROW_IE_EXCEPTION " << parse_result.error_msg << std::endl;
            THROW_IE_EXCEPTION << parse_result.error_msg;
        }

        pugi::xml_document& xmlDoc = *parse_result.xml;

        using namespace XMLParseUtils;
        pugi::xml_node ieNode = xmlDoc.document_element();
        pugi::xml_node devicesNode = ieNode.child("plugins");

        for (auto pluginNode = devicesNode.child("plugin"); !pluginNode.empty();
             pluginNode = pluginNode.next_sibling("plugin")) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:               pluginNode = pluginNode.next_sibling('plugin')) {" << std::endl;
            std::string deviceName = GetStrAttr(pluginNode, "name");
            FileUtils::FilePath pluginPath = FileUtils::toFilePath(GetStrAttr(pluginNode, "location").c_str());

            if (deviceName.find('.') != std::string::npos) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:              if (deviceName.find('.') != std::string::npos) {" << std::endl;
                THROW_IE_EXCEPTION << "Device name must not contain dot '.' symbol";
            }
            // append IR library path for default IE plugins
            {
                std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry: add empty path" << std::endl;
                //FileUtils::FilePath absFilePath = FileUtils::makePath(getInferenceEngineLibraryPath(), pluginPath);
                FileUtils::FilePath absFilePath = FileUtils::makePath(FileUtils::toFilePath(""), pluginPath);
                //if (FileUtils::fileExist(absFilePath)) pluginPath = absFilePath;
                pluginPath = absFilePath;
            }

            // check properties
            auto propertiesNode = pluginNode.child("properties");
            std::map<std::string, std::string> config;

            if (propertiesNode) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:              if (propertiesNode) {" << std::endl;
                for (auto propertyNode = propertiesNode.child("property"); !propertyNode.empty();
                     propertyNode = propertyNode.next_sibling("property")) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:                       propertyNode = propertyNode.next_sibling('property')) {" << std::endl;
                    std::string key = GetStrAttr(propertyNode, "key");
                    std::string value = GetStrAttr(propertyNode, "value");
                    config[key] = value;
                }
            }

            // check extensions
            auto extensionsNode = pluginNode.child("extensions");
            std::vector<FileUtils::FilePath> listOfExtentions;

            if (extensionsNode) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:              if (extensionsNode) {" << std::endl;
                for (auto extensionNode = extensionsNode.child("extension"); !extensionNode.empty();
                     extensionNode = extensionNode.next_sibling("extension")) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:                       extensionNode = extensionNode.next_sibling('extension')) {" << std::endl;
                    FileUtils::FilePath extensionLocation = FileUtils::toFilePath(GetStrAttr(extensionNode, "location").c_str());
                    listOfExtentions.push_back(extensionLocation);
                }
            }

            // fill value in plugin registry for later lazy initialization
            {
                std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry: PluginDescriptor" << std::endl;
                PluginDescriptor desc = {pluginPath, config, listOfExtentions};
                pluginRegistry[deviceName] = desc;
            }
        }
        std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry: end" << std::endl;
    }

    //
    // ICore public API
    //

    /**
     * @brief Returns global task executor
     * @return Reference to task executor
     */
    ITaskExecutor::Ptr GetTaskExecutor() const override {
        return _taskExecutor;
    }

    IE_SUPPRESS_DEPRECATED_START

    /**
     * @brief Returns reference to plugin by a device name
     * @param deviceName - a name of device
     * @return Reference to a plugin
     */
    InferenceEnginePluginPtr GetPluginByName(const std::string& deviceName) const override {
        return static_cast<InferenceEnginePluginPtr>(GetCPPPluginByName(deviceName));
    }

    /**
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName - a name of device
     * @return Reference to a CPP plugin wrapper
     */
    InferencePlugin GetCPPPluginByName(const std::string& deviceName) const {
        IE_SUPPRESS_DEPRECATED_START
        std::cerr << "dldt ie_core.cpp GetCPPPluginByName: begin" << std::endl;
        auto it = pluginRegistry.find(deviceName);
        if (it == pluginRegistry.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (it == pluginRegistry.end()) {" << std::endl;
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // Plugin is in registry, but not created, let's create
        std::cerr << "dldt ie_core.cpp GetCPPPluginByName: Plugin is in registry, but not created, let's create" << std::endl;
        if (plugins.find(deviceName) == plugins.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (plugins.find(deviceName) == plugins.end()) {" << std::endl;
            std::cerr << "dldt ie_core.cpp GetCPPPluginByName: it->second" << std::endl;
            PluginDescriptor desc = it->second;

            try {
                std::cerr << "dldt ie_core.cpp GetCPPPluginByName: try desc.libraryLocation" << std::endl;
                InferenceEnginePluginPtr plugin(desc.libraryLocation);
                std::cerr << "dldt ie_core.cpp GetCPPPluginByName: try plugin.operator->()" << std::endl;
                IInferencePlugin* pplugin = static_cast<IInferencePlugin*>(plugin.operator->());
                IInferencePluginAPI* iplugin_api_ptr = dynamic_cast<IInferencePluginAPI*>(pplugin);

                if (iplugin_api_ptr != nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:                  if (iplugin_api_ptr != nullptr) {" << std::endl;
                    iplugin_api_ptr->SetName(deviceName);

                    // Set Inference Engine class reference to plugins
                    ICore* mutableCore = const_cast<ICore*>(static_cast<const ICore*>(this));
                    iplugin_api_ptr->SetCore(mutableCore);
                }

                InferencePlugin cppPlugin(plugin);

                // configuring
                {
                    std::cerr << "dldt ie_core.cpp GetCPPPluginByName: configuring" << std::endl;
                    cppPlugin.SetConfig(desc.defaultConfig);

                    for (auto&& extensionLocation : desc.listOfExtentions) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:                      for (auto&& extensionLocation : desc.listOfExtentions) {" << std::endl;
                        // TODO: fix once InferenceEngine::Extension can accept FileUtils::FilePath
                        // currently, extensions cannot be loaded using wide path
                        cppPlugin.AddExtension(make_so_pointer<IExtension>(FileUtils::fromFilePath(extensionLocation)));
                    }

                    if (listener) plugin->SetLogCallback(*listener);
                }

                plugins[deviceName] = cppPlugin;
            } catch (const details::InferenceEngineException& ex) {
                std::cerr << "dldt ie_core.cpp GetCPPPluginByName: THROW_IE_EXCEPTION Failed to create plugin" << std::endl;
                THROW_IE_EXCEPTION << "Failed to create plugin " << FileUtils::fromFilePath(desc.libraryLocation) << " for device " << deviceName
                                   << "\n"
                                   << "Please, check your environment\n"
                                   << ex.what() << "\n";
            }
        }

        IE_SUPPRESS_DEPRECATED_END

        return plugins[deviceName];
    }

    IE_SUPPRESS_DEPRECATED_END

    /**
     * @brief Unregisters plugin for specified device
     * @param deviceName - a name of device
     */
    void UnregisterPluginByName(const std::string& deviceName) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      void UnregisterPluginByName(const std::string& deviceName) {" << std::endl;
        auto it = plugins.find(deviceName);
        if (it == plugins.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (it == plugins.end()) {" << std::endl;
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        plugins.erase(deviceName);
    }

    /**
     * @brief Registers plugin in registry for specified device
     * @param deviceName - a name of device
     */
    void RegisterPluginByName(const std::string& pluginName, const std::string& deviceName) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      void RegisterPluginByName(const std::string& pluginName, const std::string& deviceName) {" << std::endl;
        auto it = pluginRegistry.find(deviceName);
        if (it != pluginRegistry.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (it != pluginRegistry.end()) {" << std::endl;
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\"  is already registered in the InferenceEngine";
        }

        if (deviceName.find('.') != std::string::npos) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (deviceName.find('.') != std::string::npos) {" << std::endl;
            THROW_IE_EXCEPTION << "Device name must not contain dot '.' symbol";
        }

        // append IR library path for default IE plugins
        FileUtils::FilePath pluginPath;
        {
            pluginPath = FileUtils::makeSharedLibraryName({}, FileUtils::toFilePath(pluginName.c_str()));

            FileUtils::FilePath absFilePath = FileUtils::makePath(getInferenceEngineLibraryPath(), pluginPath);
            if (FileUtils::fileExist(absFilePath)) pluginPath = absFilePath;
        }

        PluginDescriptor desc = {pluginPath, {}, {}};
        pluginRegistry[deviceName] = desc;
    }

    std::vector<std::string> GetListOfDevicesInRegistry() const {
        std::vector<std::string> listOfDevices;
        for (auto&& pluginDesc : pluginRegistry) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          for (auto&& pluginDesc : pluginRegistry) {" << std::endl;
            listOfDevices.push_back(pluginDesc.first);
        }

        return listOfDevices;
    }

    void SetConfigForPlugins(const std::map<std::string, std::string>& config, const std::string& deviceName) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      void SetConfigForPlugins(const std::map<std::string, std::string>& config, const std::string& deviceName) {" << std::endl;
        std::cerr << "dldt ie_core.cpp SetConfigForPlugins: begin" << std::endl;
        // set config for plugins in registry
        for (auto& desc : pluginRegistry) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          for (auto& desc : pluginRegistry) {" << std::endl;
            if (deviceName.empty() || deviceName == desc.first) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:              if (deviceName.empty() || deviceName == desc.first) {" << std::endl;
                for (auto&& conf : config) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:                  for (auto&& conf : config) {" << std::endl;
                    desc.second.defaultConfig[conf.first] = conf.second;
                }
            }
        }

        // set config for already created plugins
        for (auto& plugin : plugins) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          for (auto& plugin : plugins) {" << std::endl;
            if (deviceName.empty() || deviceName == plugin.first) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:              if (deviceName.empty() || deviceName == plugin.first) {" << std::endl;
                IE_SUPPRESS_DEPRECATED_START
                plugin.second.SetConfig(config);
                IE_SUPPRESS_DEPRECATED_END
            }
        }
    }

    void SetErrorListener(IErrorListener* list) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      void SetErrorListener(IErrorListener* list) {" << std::endl;
        listener = list;

        // set for already created plugins
        for (auto& plugin : plugins) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          for (auto& plugin : plugins) {" << std::endl;
            IE_SUPPRESS_DEPRECATED_START
            GetPluginByName(plugin.first)->SetLogCallback(*listener);
            IE_SUPPRESS_DEPRECATED_END
        }
    }

    void addExtension(const IExtensionPtr& extension) {
        std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp: void addExtension(const IExtensionPtr& extension) {" << std::endl;
        /*std::map<std::string, ngraph::OpSet> opsets;
        try {
            std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp: try { opsets = extension->getOpSets();" << std::endl;
            opsets = extension->getOpSets();
            std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp: getOpSets success" << std::endl;
        } catch (...) {
            std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp: } catch (...) {" << std::endl;}
        for (const auto& it : opsets) {
            std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp: for (const auto& it : opsets) {" << std::endl;
            if (opsetNames.find(it.first) != opsetNames.end()) {
                std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp: if (opsetNames.find(it.first) != opsetNames.end()) {" << std::endl;
                THROW_IE_EXCEPTION << "Cannot add opset with name: " << it.first << ". Opset with the same name already exists.";
            }
            std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp: opsetNames.insert(it.first);" << std::endl;
            opsetNames.insert(it.first);
        }*/
        std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp: extensions.emplace_back(extension);" << std::endl;
        extensions.emplace_back(extension);
        std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp: addExtension end" << std::endl;
    }

    const std::vector<IExtensionPtr>& getExtensions() {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      const std::vector<IExtensionPtr>& getExtensions() {" << std::endl;
        return extensions;
    }
};

Core::Impl::Impl() {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  Core::Impl::Impl() {" << std::endl;
    opsetNames.insert("opset1");
}

Core::Impl::~Impl() {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  Core::Impl::~Impl() {" << std::endl;}

Core::Core(const std::string& xmlConfigFile) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  Core::Core(const std::string& xmlConfigFile) {" << std::endl;
    _impl = std::make_shared<Impl>();

    std::string xmlConfigFile_ = xmlConfigFile;
    if (xmlConfigFile_.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (xmlConfigFile_.empty()) {" << std::endl;
        // register plugins from default plugins.xml config
        //FileUtils::FilePath xmlConfigFileDefault = FileUtils::makePath(getInferenceEngineLibraryPath(), FileUtils::toFilePath("plugins.xml"));
        FileUtils::FilePath xmlConfigFileDefault = FileUtils::makePath(FileUtils::toFilePath("."), FileUtils::toFilePath("plugins.xml"));
        xmlConfigFile_ = FileUtils::fromFilePath(xmlConfigFileDefault);
    }

    RegisterPlugins(xmlConfigFile_);
}

std::map<std::string, Version> Core::GetVersions(const std::string& deviceName) const {
    std::map<std::string, Version> versions;
    std::vector<std::string> deviceNames;

    {
        // for compatibility with samples / demo
        if (deviceName.find("HETERO:") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (deviceName.find('HETERO:') == 0) {" << std::endl;
            deviceNames = DeviceIDParser::getHeteroDevices(deviceName.substr(7));
            deviceNames.push_back("HETERO");
        } else if (deviceName.find("MULTI") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          } else if (deviceName.find('MULTI') == 0) {" << std::endl;
            deviceNames.push_back("MULTI");
            deviceNames = DeviceIDParser::getMultiDevices(deviceName.substr(6));
        } else {
            deviceNames.push_back(deviceName);
        }
    }

    for (auto&& deviceName_ : deviceNames) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      for (auto&& deviceName_ : deviceNames) {" << std::endl;
        DeviceIDParser parser(deviceName_);
        std::string deviceNameLocal = parser.getDeviceName();

        IE_SUPPRESS_DEPRECATED_START
        const Version* version = _impl->GetCPPPluginByName(deviceNameLocal).GetVersion();
        IE_SUPPRESS_DEPRECATED_END
        versions[deviceNameLocal] = *version;
    }

    return versions;
}

void Core::SetLogCallback(IErrorListener& listener) const {
    _impl->SetErrorListener(&listener);
}

namespace {
template <typename T>
struct Parsed {
    std::string _deviceName;
    std::map<std::string, T> _config;
};

template <typename T = Parameter>
Parsed<T> parseDeviceNameIntoConfig(const std::string& deviceName, const std::map<std::string, T>& config = {}) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  Parsed<T> parseDeviceNameIntoConfig(const std::string& deviceName, const std::map<std::string, T>& config = {}) {" << std::endl;
    auto config_ = config;
    auto deviceName_ = deviceName;
    if (deviceName_.find("HETERO:") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName_.find('HETERO:') == 0) {" << std::endl;
        deviceName_ = "HETERO";
        config_["TARGET_FALLBACK"] = deviceName.substr(7);
    } else if (deviceName_.find("MULTI:") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      } else if (deviceName_.find('MULTI:') == 0) {" << std::endl;
        deviceName_ = "MULTI";
        config_[InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = deviceName.substr(6);
    } else {
        DeviceIDParser parser(deviceName_);
        deviceName_ = parser.getDeviceName();
        std::string deviceIDLocal = parser.getDeviceID();

        if (!deviceIDLocal.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (!deviceIDLocal.empty()) {" << std::endl;
            config_[KEY_DEVICE_ID] = deviceIDLocal;
        }
    }
    return {deviceName_, config_};
}
}  //  namespace

CNNNetwork Core::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    IE_PROFILING_AUTO_SCOPE(Core::ReadNetwork)
    IE_SUPPRESS_DEPRECATED_START
    auto cnnReader = std::shared_ptr<ICNNNetReader>(CreateCNNNetReader());
    ResponseDesc desc;
    StatusCode rt = cnnReader->ReadNetwork(modelPath.c_str(), &desc);
    if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
#if defined(ENABLE_NGRAPH)
    auto cnnNetReaderImpl = std::dynamic_pointer_cast<details::CNNNetReaderImpl>(cnnReader);
    if (cnnNetReaderImpl && cnnReader->getVersion(&desc) >= 10) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (cnnNetReaderImpl && cnnReader->getVersion(&desc) >= 10) {" << std::endl;
        cnnNetReaderImpl->addExtensions(_impl->getExtensions());
    }
#endif
    std::string bPath = binPath;
    if (bPath.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (bPath.empty()) {" << std::endl;
        bPath = modelPath;
        auto pos = bPath.rfind('.');
        if (pos != std::string::npos) bPath = bPath.substr(0, pos);
        bPath += ".bin";

        if (!FileUtils::fileExist(bPath)) bPath.clear();
    }

    if (!bPath.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (!bPath.empty()) {" << std::endl;
        rt = cnnReader->ReadWeights(bPath.c_str(), &desc);
        if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
    } else {
        TBlob<uint8_t>::Ptr weights_ptr;
        rt = cnnReader->SetWeights(weights_ptr, &desc);
        if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
    }
    IE_SUPPRESS_DEPRECATED_END

    return CNNNetwork(cnnReader);
}

CNNNetwork Core::ReadNetwork(const std::string& model, const Blob::CPtr& weights) const {
    IE_PROFILING_AUTO_SCOPE(Core::ReadNetwork)
    IE_SUPPRESS_DEPRECATED_START
    auto cnnReader = std::shared_ptr<ICNNNetReader>(CreateCNNNetReader());
    ResponseDesc desc;
    StatusCode rt = cnnReader->ReadNetwork(model.data(), model.length(), &desc);
    if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
#if defined(ENABLE_NGRAPH)
    auto cnnNetReaderImpl = std::dynamic_pointer_cast<details::CNNNetReaderImpl>(cnnReader);
    if (cnnNetReaderImpl && cnnReader->getVersion(&desc) >= 10) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (cnnNetReaderImpl && cnnReader->getVersion(&desc) >= 10) {" << std::endl;
        cnnNetReaderImpl->addExtensions(_impl->getExtensions());
    }
#endif
    TBlob<uint8_t>::Ptr weights_ptr;
    if (weights) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (weights) {" << std::endl;
        uint8_t* ptr = weights->cbuffer().as<uint8_t*>();
        weights_ptr = make_shared_blob<uint8_t>(weights->getTensorDesc(), ptr);
    }
    rt = cnnReader->SetWeights(weights_ptr, &desc);
    if (rt != OK) THROW_IE_EXCEPTION << desc.msg;
    IE_SUPPRESS_DEPRECATED_END
    return CNNNetwork(cnnReader);
}

ExecutableNetwork Core::LoadNetwork(CNNNetwork network, const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:                                      const std::map<std::string, std::string>& config) {" << std::endl;
    IE_PROFILING_AUTO_SCOPE(Core::LoadNetwork)
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    IE_SUPPRESS_DEPRECATED_START
    return _impl->GetCPPPluginByName(parsed._deviceName).LoadNetwork(network, parsed._config);
    IE_SUPPRESS_DEPRECATED_END
}

void Core::AddExtension(const IExtensionPtr& extension) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  void Core::AddExtension(const IExtensionPtr& extension) {" << std::endl;
    _impl->addExtension(extension);
}

ExecutableNetwork Core::LoadNetwork(CNNNetwork network, RemoteContext::Ptr context,
                                    const std::map<std::string, std::string>& config) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:                                      const std::map<std::string, std::string>& config) {" << std::endl;
    IE_PROFILING_AUTO_SCOPE(Core::LoadNetwork)
    std::map<std::string, std::string> config_ = config;

    std::string deviceName_ = context->getDeviceName();
    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(deviceName));

    if (pluginAPIInterface == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (pluginAPIInterface == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << deviceName << " does not implement the LoadNetwork method";
    }

    return pluginAPIInterface->LoadNetwork(network, config_, context);
    IE_SUPPRESS_DEPRECATED_END
}

RemoteContext::Ptr Core::CreateContext(const std::string& deviceName_, const ParamMap& params) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  RemoteContext::Ptr Core::CreateContext(const std::string& deviceName_, const ParamMap& params) {" << std::endl;
    if (deviceName_.find("HETERO") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName_.find('HETERO') == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "HETERO device does not support remote contexts";
    }
    if (deviceName_.find("MULTI") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName_.find('MULTI') == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "MULTI device does not support remote contexts";
    }

    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(deviceName));

    if (pluginAPIInterface == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (pluginAPIInterface == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << deviceName << " does not implement the CreateContext method";
    }

    return pluginAPIInterface->CreateContext(params);
    IE_SUPPRESS_DEPRECATED_END
}

RemoteContext::Ptr Core::GetDefaultContext(const std::string& deviceName_) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  RemoteContext::Ptr Core::GetDefaultContext(const std::string& deviceName_) {" << std::endl;
    if (deviceName_.find("HETERO") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName_.find('HETERO') == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "HETERO device does not support remote contexts";
    }
    if (deviceName_.find("MULTI") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName_.find('MULTI') == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "MULTI device does not support remote contexts";
    }

    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();

    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(deviceName));

    if (pluginAPIInterface == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (pluginAPIInterface == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << deviceName << " does not implement the CreateContext method";
    }

    return pluginAPIInterface->GetDefaultContext();
    IE_SUPPRESS_DEPRECATED_END
}

void Core::AddExtension(IExtensionPtr extension, const std::string& deviceName_) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  void Core::AddExtension(IExtensionPtr extension, const std::string& deviceName_) {" << std::endl;
    if (deviceName_.find("HETERO") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName_.find('HETERO') == 0) {" << std::endl;
        THROW_IE_EXCEPTION
            << "HETERO device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("MULTI") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName_.find('MULTI') == 0) {" << std::endl;
        THROW_IE_EXCEPTION
            << "MULTI device does not support extensions. Please, set extensions directly to fallback devices";
    }

    DeviceIDParser parser(deviceName_);
    std::string deviceName = parser.getDeviceName();

    IE_SUPPRESS_DEPRECATED_START
    _impl->GetCPPPluginByName(deviceName).AddExtension(extension);
    _impl->addExtension(extension);
    IE_SUPPRESS_DEPRECATED_END
}

ExecutableNetwork Core::ImportNetwork(const std::string& modelFileName, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:                                        const std::map<std::string, std::string>& config) {" << std::endl;
    if (deviceName.find("HETERO") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName.find('HETERO') == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "HETERO device does not support ImportNetwork";
    }
    if (deviceName.find("MULTI") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName.find('MULTI') == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "MULTI device does not support ImportNetwork";
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName, config);

    IE_SUPPRESS_DEPRECATED_START
    return _impl->GetCPPPluginByName(parsed._deviceName).ImportNetwork(modelFileName, parsed._config);
    IE_SUPPRESS_DEPRECATED_END
}

IE_SUPPRESS_DEPRECATED_START

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:                                        const std::map<std::string, std::string>& config) {" << std::endl;
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);

    if (parsed._deviceName.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (parsed._deviceName.empty()) {" << std::endl;
        ExportMagic magic = {};
        networkModel.read(magic.data(), magic.size());
        auto exportedWithName = (exportMagic == magic);
        if (exportedWithName) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (exportedWithName) {" << std::endl;
            std::getline(networkModel, parsed._deviceName);
        }
        networkModel.seekg(0, networkModel.beg);
    }

    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(parsed._deviceName));
    if (pluginAPIInterface == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (pluginAPIInterface == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << parsed._deviceName << " does not implement the ImportNetwork method";
    }

    return pluginAPIInterface->ImportNetwork(networkModel, parsed._config);
}

IE_SUPPRESS_DEPRECATED_END

QueryNetworkResult Core::QueryNetwork(const ICNNNetwork& network, const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) const {
    if (deviceName.find("MULTI") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName.find('MULTI') == 0) {" << std::endl;
        THROW_IE_EXCEPTION << "MULTI device does not support QueryNetwork";
    }

    QueryNetworkResult res;
    auto parsed = parseDeviceNameIntoConfig(deviceName, config);
    IE_SUPPRESS_DEPRECATED_START
    _impl->GetCPPPluginByName(parsed._deviceName).QueryNetwork(network, parsed._config, res);
    IE_SUPPRESS_DEPRECATED_END
    return res;
}

void Core::SetConfig(const std::map<std::string, std::string>& config, const std::string& deviceName) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  void Core::SetConfig(const std::map<std::string, std::string>& config, const std::string& deviceName) {" << std::endl;
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (deviceName.find('HETERO:') == 0) {" << std::endl;
            THROW_IE_EXCEPTION << "SetConfig is supported only for HETERO itself (without devices). "
                                  "You can configure the devices with SetConfig before creating the HETERO on top.";
        }

        if (config.find("TARGET_FALLBACK") != config.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (config.find('TARGET_FALLBACK') != config.end()) {" << std::endl;
            THROW_IE_EXCEPTION << "Please, specify TARGET_FALLBACK to the LoadNetwork directly, "
                                  "as you will need to pass the same TARGET_FALLBACK anyway.";
        }
    }

    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (deviceName.find('MULTI:') == 0) {" << std::endl;
            THROW_IE_EXCEPTION << "SetConfig is supported only for MULTI itself (without devices). "
                                  "You can configure the devices with SetConfig before creating the MULTI on top.";
        }

        if (config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES) != config.end()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES) != config.end()) {" << std::endl;
            THROW_IE_EXCEPTION << "Please, specify DEVICE_PRIORITIES to the LoadNetwork directly, "
                                  "as you will need to pass the same DEVICE_PRIORITIES anyway.";
        }
    }

    if (deviceName.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (deviceName.empty()) {" << std::endl;
        _impl->SetConfigForPlugins(config, std::string());
    } else {
        auto parsed = parseDeviceNameIntoConfig(deviceName, config);
        _impl->SetConfigForPlugins(parsed._config, parsed._deviceName);
    }
}

Parameter Core::GetConfig(const std::string& deviceName, const std::string& name) const {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (deviceName.find('HETERO:') == 0) {" << std::endl;
            THROW_IE_EXCEPTION
                << "You can only GetConfig of the HETERO itself (without devices). "
                   "GetConfig is also possible for the individual devices before creating the HETERO on top.";
        }
    }
    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (deviceName.find('MULTI:') == 0) {" << std::endl;
            THROW_IE_EXCEPTION
                << "You can only GetConfig of the MULTI itself (without devices). "
                   "GetConfig is also possible for the individual devices before creating the MULTI on top.";
        }
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName);
    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(parsed._deviceName));
    IE_SUPPRESS_DEPRECATED_END
    if (pluginAPIInterface == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (pluginAPIInterface == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << parsed._deviceName << " does not implement the GetConfig method";
    }
    return pluginAPIInterface->GetConfig(name, parsed._config);
}

Parameter Core::GetMetric(const std::string& deviceName, const std::string& name) const {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (deviceName.find('HETERO:') == 0) {" << std::endl;
            THROW_IE_EXCEPTION
                << "You can get specific metrics with the GetMetric only for the HETERO itself (without devices). "
                   "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (deviceName.find('MULTI:') == 0) {" << std::endl;
            THROW_IE_EXCEPTION
                << "You can get specific metrics with the GetMetric only for the MULTI itself (without devices). "
                   "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    auto parsed = parseDeviceNameIntoConfig(deviceName);
    IE_SUPPRESS_DEPRECATED_START
    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(parsed._deviceName));
    IE_SUPPRESS_DEPRECATED_END
    if (pluginAPIInterface == nullptr) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      if (pluginAPIInterface == nullptr) {" << std::endl;
        THROW_IE_EXCEPTION << parsed._deviceName << " does not implement the GetMetric method";
    }

    return pluginAPIInterface->GetMetric(name, parsed._config);
}

std::vector<std::string> Core::GetAvailableDevices() const {
    std::vector<std::string> devices;

    std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto&& deviceName : _impl->GetListOfDevicesInRegistry()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:      for (auto&& deviceName : _impl->GetListOfDevicesInRegistry()) {" << std::endl;
        Parameter p;
        std::vector<std::string> devicesIDs;

        try {
            p = GetMetric(deviceName, propertyName);
            devicesIDs = p.as<std::vector<std::string>>();
        } catch (details::InferenceEngineException&) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          } catch (details::InferenceEngineException&) {" << std::endl;
            // plugin is not created by e.g. invalid env
        } catch (const std::exception& ex) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          } catch (const std::exception& ex) {" << std::endl;
            THROW_IE_EXCEPTION << "An exception is thrown while trying to create the " << deviceName
                               << " device and call GetMetric: " << ex.what();
        } catch (...) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          } catch (...) {" << std::endl;
            THROW_IE_EXCEPTION << "Unknown exception is thrown while trying to create the " << deviceName
                               << " device and call GetMetric";
        }

        if (devicesIDs.size() > 1) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          if (devicesIDs.size() > 1) {" << std::endl;
            for (auto&& deviceID : devicesIDs) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:              for (auto&& deviceID : devicesIDs) {" << std::endl;
                devices.push_back(deviceName + '.' + deviceID);
            }
        } else if (!devicesIDs.empty()) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:          } else if (!devicesIDs.empty()) {" << std::endl;
            devices.push_back(deviceName);
        }
    }

    return devices;
}

void Core::RegisterPlugin(const std::string& pluginName, const std::string& deviceName) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  void Core::RegisterPlugin(const std::string& pluginName, const std::string& deviceName) {" << std::endl;
    _impl->RegisterPluginByName(pluginName, deviceName);
}

void Core::RegisterPlugins(const std::string& xmlConfigFile) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  void Core::RegisterPlugins(const std::string& xmlConfigFile) {" << std::endl;
    _impl->RegisterPluginsInRegistry(xmlConfigFile);
}

void Core::UnregisterPlugin(const std::string& deviceName_) {
    std::cerr << "./inference-engine/src/inference_engine/ie_core.cpp:  void Core::UnregisterPlugin(const std::string& deviceName_) {" << std::endl;
    DeviceIDParser parser(deviceName_);
    std::string deviceName = parser.getDeviceName();

    _impl->UnregisterPluginByName(deviceName);
}

}  // namespace InferenceEngine
