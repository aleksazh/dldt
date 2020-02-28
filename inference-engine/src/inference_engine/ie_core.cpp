// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core.hpp"
#include "ie_plugin_config.hpp"
#include "details/caseless.hpp"
#include "details/ie_exception_conversion.hpp"
#include "cpp_interfaces/base/ie_plugin_base.hpp"
#include "details/ie_so_pointer.hpp"
#include "multi-device/multi_device_config.hpp"

#include "ie_util_internal.hpp"
#include "file_utils.h"
#include "ie_icore.hpp"

#include <fstream>
#include <sstream>
#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <map>
#include <iostream>

#include "xml_parse_utils.h"

file_name_t make_plugin_name(const file_name_t &path, const file_name_t &input);

using namespace InferenceEngine::PluginConfigParams;

namespace InferenceEngine {

namespace  {

IInferencePluginAPI * getInferencePluginAPIInterface(IInferencePlugin * iplugin) {
    return dynamic_cast<IInferencePluginAPI *>(iplugin);
}

IInferencePluginAPI * getInferencePluginAPIInterface(InferenceEnginePluginPtr iplugin) {
    return getInferencePluginAPIInterface(static_cast<IInferencePlugin *>(iplugin.operator->()));
}

IInferencePluginAPI * getInferencePluginAPIInterface(InferencePlugin plugin) {
    return getInferencePluginAPIInterface(static_cast<InferenceEnginePluginPtr>(plugin));
}

}  // namespace

DeviceIDParser::DeviceIDParser(const std::string & deviceNameWithID) {
    deviceName = deviceNameWithID;

    auto pos = deviceName.find('.');
    if (pos != std::string::npos) {
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
    std::vector<std::string> deviceNames;

    std::string cdevice;
    char delimiter = ',';
    size_t pos = 0;

    while ((pos = fallbackDevice.find(delimiter)) != std::string::npos) {
        deviceNames.push_back(fallbackDevice.substr(0, pos));
        fallbackDevice.erase(0, pos + 1);
    }

    if (!fallbackDevice.empty())
        deviceNames.push_back(fallbackDevice);

    return deviceNames;
}

std::vector<std::string> DeviceIDParser::getMultiDevices(std::string devicesList) {
    std::vector<std::string> deviceNames;
    auto trim_request_info = [] (std::string device_with_requests){
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    std::string device;
    char delimiter = ',';
    size_t pos = 0;
    // in addition to the list of devices, every device can have a #requests in the brackets e.g. "CPU(100)"
    // we skip the #requests info here
    while ((pos = devicesList.find(delimiter)) != std::string::npos) {
        auto d = devicesList.substr(0, pos);
        deviceNames.push_back(trim_request_info(d));
        devicesList.erase(0, pos + 1);
    }

    if (!devicesList.empty())
        deviceNames.push_back(trim_request_info(devicesList));

    return deviceNames;
}


class Core::Impl : public ICore {
    ITaskExecutor::Ptr          _taskExecutor = nullptr;
    mutable std::map<std::string, InferencePlugin, details::CaselessLess<std::string> > plugins;

    struct PluginDescriptor {
        file_name_t libraryLocation;
        std::map<std::string, std::string> defaultConfig;
        std::vector<std::string> listOfExtentions;
    };
    std::map<std::string, PluginDescriptor, details::CaselessLess<std::string> > pluginRegistry;
    IErrorListener * listener = nullptr;

public:
    ~Impl() override;

    /**
     * @brief Register plugins for devices which are located in .xml configuration file
     * @param xmlConfigFile - an .xml configuraion with device / plugin information
     */
    void RegisterPluginsInRegistry(const std::string & xmlConfigFile) {
        std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry" << std::endl;
        pugi::xml_document xmlDoc;
        std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 1" << std::endl;
        pugi::xml_parse_result res = xmlDoc.load_file(xmlConfigFile.c_str());

        std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 2" << std::endl;
        if (res.status != pugi::status_ok) {
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 3" << std::endl;
            std::ifstream t(xmlConfigFile);
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 4" << std::endl;
            std::string str((std::istreambuf_iterator<char>(t)),
                            std::istreambuf_iterator<char>());
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 5" << std::endl;

            int line = 1;
            int pos = 0;
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 6" << std::endl;
            for (auto && token : str) {
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 7" << std::endl;
                if (token == '\n') {
                    line++;
                    pos = 0;
                } else {
                    pos++;
                }
                if (pos >= res.offset) {
                    break;
                }
            }

            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 8" << std::endl;
            THROW_IE_EXCEPTION << "Error loading xmlfile: " << xmlConfigFile << ", " << res.description()
                                           << " at line: " << line << " pos: " << pos;
        }

        std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 9" << std::endl;
        using namespace XMLParseUtils;
        pugi::xml_node ieNode = xmlDoc.document_element();
        std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 10" << std::endl;
        pugi::xml_node devicesNode = ieNode.child("plugins");

        for (auto pluginNode = devicesNode.child("plugin"); !pluginNode.empty();
             pluginNode = pluginNode.next_sibling("plugin")) {
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 11" << std::endl;
            std::string deviceName = GetStrAttr(pluginNode, "name");
            file_name_t pluginPath = GetStrAttr(pluginNode, "location");

            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 12" << std::endl;
            if (deviceName.find('.') != std::string::npos) {
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 13" << std::endl;
                THROW_IE_EXCEPTION << "Device name must not contain dot '.' symbol";
            }

            // append IR library path for default IE plugins
            {
                std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 14" << std::endl;
                std::string absPluginPath = FileUtils::makePath(getIELibraryPath(), pluginPath);
                if (FileUtils::fileExist(absPluginPath)) {
                    std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 15" << std::endl;
                    pluginPath = absPluginPath;
                }
            }

            // check properties
            auto propertiesNode = pluginNode.child("properties");
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 16" << std::endl;
            std::map<std::string, std::string> config;

            if (propertiesNode) {
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 17" << std::endl;
                for (auto propertyNode = propertiesNode.child("property"); !propertyNode.empty();
                     propertyNode = propertyNode.next_sibling("property")) {
                    std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 18" << std::endl;
                    std::string key = GetStrAttr(propertyNode, "key");
                    std::string value = GetStrAttr(propertyNode, "value");
                    std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 19" << std::endl;
                    config[key] = value;
                }
            }

            // check extensions
            auto extensionsNode = pluginNode.child("extensions");
            std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 20" << std::endl;
            std::vector<std::string> listOfExtentions;

            if (extensionsNode) {
                std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 21" << std::endl;
                for (auto extensionNode = extensionsNode.child("extension"); !extensionNode.empty();
                     extensionNode = extensionNode.next_sibling("extension")) {
                    std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 22" << std::endl;
                    std::string extensionLocation = GetStrAttr(extensionNode, "location");
                    std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 23" << std::endl;
                    listOfExtentions.push_back(extensionLocation);
                }
            }

            // fill value in plugin registry for later lazy initialization
            {
                PluginDescriptor desc = { pluginPath, config, listOfExtentions };
                std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 24" << std::endl;
                pluginRegistry[deviceName] = desc;
                std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry 25" << std::endl;
            }
        }
        std::cerr << "dldt ie_core.cpp RegisterPluginsInRegistry the end" << std::endl;
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

    /**
     * @brief Returns reference to plugin by a device name
     * @param deviceName - a name of device
     * @return Reference to a plugin
     */
    InferenceEnginePluginPtr GetPluginByName(const std::string & deviceName) const override {
        return static_cast<InferenceEnginePluginPtr>(GetCPPPluginByName(deviceName));
    }

    /**
     * @brief Returns reference to CPP plugin wrapper by a device name
     * @param deviceName - a name of device
     * @return Reference to a CPP plugin wrapper
     */
    InferencePlugin GetCPPPluginByName(const std::string & deviceName) const {
        std::cerr << "dldt ie_core.cpp GetCPPPluginByName" << std::endl;
        auto it = pluginRegistry.find(deviceName);
        for (auto && pluginDesc : pluginRegistry) {
            std::cerr << "dldt ie_core.cpp GetCPPPluginByName device: " << pluginDesc.first << std::endl;
        }
        std::cerr << "dldt ie_core.cpp GetCPPPluginByName 1" << std::endl;
        if (it == pluginRegistry.end()) {
        std::cerr << "dldt ie_core.cpp GetCPPPluginByName 2 deviceName:" << deviceName << std::endl;
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        // Plugin is in registry, but not created, let's create

        std::cerr << "dldt ie_core.cpp GetCPPPluginByName 3" << std::endl;
        if (plugins.find(deviceName) == plugins.end()) {
            std::cerr << "dldt ie_core.cpp GetCPPPluginByName 4" << std::endl;
            PluginDescriptor desc = it->second;

            std::cerr << "dldt ie_core.cpp GetCPPPluginByName 5" << std::endl;
            try {
                std::cerr << "dldt ie_core.cpp GetCPPPluginByName 6" << std::endl;
                InferenceEnginePluginPtr plugin(desc.libraryLocation);
                std::cerr << "dldt ie_core.cpp GetCPPPluginByName 7" << std::endl;
                IInferencePlugin * pplugin = static_cast<IInferencePlugin *>(plugin.operator->());
                std::cerr << "dldt ie_core.cpp GetCPPPluginByName 8" << std::endl;
                IInferencePluginAPI * iplugin_api_ptr = dynamic_cast<IInferencePluginAPI *>(pplugin);

                std::cerr << "dldt ie_core.cpp GetCPPPluginByName 9" << std::endl;
                if (iplugin_api_ptr != nullptr) {
                    std::cerr << "dldt ie_core.cpp GetCPPPluginByName 10" << std::endl;
                    iplugin_api_ptr->SetName(deviceName);

                    // Set Inference Engine class reference to plugins
                    ICore * mutableCore = const_cast<ICore *>(static_cast<const ICore *>(this));
                    iplugin_api_ptr->SetCore(mutableCore);
                }

                std::cerr << "dldt ie_core.cpp GetCPPPluginByName 11" << std::endl;
                InferencePlugin cppPlugin(plugin);

                // configuring
                {
                    std::cerr << "dldt ie_core.cpp GetCPPPluginByName 12" << std::endl;
                    cppPlugin.SetConfig(desc.defaultConfig);

                    for (auto && extensionLocation : desc.listOfExtentions) {
                        std::cerr << "dldt ie_core.cpp GetCPPPluginByName 13" << std::endl;
                        cppPlugin.AddExtension(make_so_pointer<IExtension>(extensionLocation));
                    }

                    if (listener)
                        std::cerr << "dldt ie_core.cpp GetCPPPluginByName 14" << std::endl;
                        plugin->SetLogCallback(*listener);
                }
                std::cerr << "dldt ie_core.cpp GetCPPPluginByName 15" << std::endl;

                plugins[deviceName] = cppPlugin;
                std::cerr << "dldt ie_core.cpp GetCPPPluginByName 16" << std::endl;
            } catch (const details::InferenceEngineException & ex) {
                std::cerr << "dldt ie_core.cpp GetCPPPluginByName 17" << std::endl;
                THROW_IE_EXCEPTION << "Failed to create plugin " << desc.libraryLocation << " for device " << deviceName << "\n"
                                   << "Please, check your environment\n"
                                   << ex.what() << "\n";
            }
        }

        std::cerr << "dldt ie_core.cpp GetCPPPluginByName 18" << std::endl;
        return plugins[deviceName];
    }

    /**
     * @brief Unregisters plugin for specified device
     * @param deviceName - a name of device
     */
    void UnregisterPluginByName(const std::string & deviceName) {
        auto it = plugins.find(deviceName);
        if (it == plugins.end()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\" name is not registered in the InferenceEngine";
        }

        plugins.erase(deviceName);
    }

    /**
     * @brief Registers plugin in registry for specified device
     * @param deviceName - a name of device
     */
    void RegisterPluginByName(const std::string & pluginName, const std::string & deviceName) {
        auto it = pluginRegistry.find(deviceName);
        if (it != pluginRegistry.end()) {
            THROW_IE_EXCEPTION << "Device with \"" << deviceName << "\"  is already registered in the InferenceEngine";
        }

        if (deviceName.find('.') != std::string::npos) {
            THROW_IE_EXCEPTION << "Device name must not contain dot '.' symbol";
        }

        // append IR library path for default IE plugins
        std::string pluginPath;
        {
            pluginPath = make_plugin_name(file_name_t(), pluginName);

            std::string absPluginPath = FileUtils::makePath(getIELibraryPath(), pluginPath);
            if (FileUtils::fileExist(absPluginPath))
                pluginPath = absPluginPath;
        }

        PluginDescriptor desc = { pluginPath, { }, { } };
        pluginRegistry[deviceName] = desc;
    }

    std::vector<std::string> GetListOfDevicesInRegistry() const {
        std::vector<std::string> listOfDevices;
        for (auto && pluginDesc : pluginRegistry) {
            listOfDevices.push_back(pluginDesc.first);
        }

        return listOfDevices;
    }

    void SetConfigForPlugins(const std::map<std::string, std::string> & config, const std::string & deviceName) {
        // set config for plugins in registry
        for (auto & desc : pluginRegistry) {
            if (deviceName.empty() || deviceName == desc.first) {
                for (auto && conf : config) {
                    desc.second.defaultConfig[conf.first] = conf.second;
                }
            }
        }

        // set config for already created plugins
        for (auto & plugin : plugins) {
            if (deviceName.empty() || deviceName == plugin.first) {
                plugin.second.SetConfig(config);
            }
        }
    }

    void SetErrorListener(IErrorListener * list) {
        listener = list;

        // set for already created plugins
        for (auto & plugin : plugins) {
            GetPluginByName(plugin.first)->SetLogCallback(*listener);
        }
    }
};

Core::Impl::~Impl() {
}

Core::Core(const std::string & xmlConfigFile) {
    std::cerr << "dldt ie_core.cpp Core::Core _impl = std::make_shared<Impl>();" << std::endl;
    _impl = std::make_shared<Impl>();

    std::cerr << "dldt ie_core.cpp Core::Core std::string xmlConfigFile_ = xmlConfigFile;" << std::endl;
    std::string xmlConfigFile_ = xmlConfigFile;
    if (xmlConfigFile_.empty()) {
        std::cerr << "dldt ie_core.cpp Core::Core xmlConfigFile is empty" << xmlConfigFile_ << std::endl;
        // register plugins from default plugins.xml config
        // Failing here
    //    xmlConfigFile_ = FileUtils::makePath(getIELibraryPath(), "plugins.xml");
    //    std::cerr << "dldt ie_core.cpp Core::Core xmlConfigFile = " << xmlConfigFile_ << std::endl;
    }

    //std::cerr << "dldt ie_core.cpp Core::Core RegisterPlugins(xmlConfigFile_);" << std::endl;
    //RegisterPlugins(xmlConfigFile_);
    //std::cerr << "dldt ie_core.cpp Core::Core the end" << std::endl;
}

std::map<std::string, Version> Core::GetVersions(const std::string & deviceName) const {
    std::map<std::string, Version> versions;
    std::vector<std::string> deviceNames;

    {
        // for compatibility with samples / demo
        if (deviceName.find("HETERO:") == 0) {
            deviceNames = DeviceIDParser::getHeteroDevices(deviceName.substr(7));
            deviceNames.push_back("HETERO");
        } else  if (deviceName.find("MULTI") == 0) {
            deviceNames.push_back("MULTI");
            deviceNames = DeviceIDParser::getMultiDevices(deviceName.substr(6));
        } else {
            deviceNames.push_back(deviceName);
        }
    }

    for (auto && deviceName_ : deviceNames) {
        DeviceIDParser parser(deviceName_);
        std::string deviceNameLocal = parser.getDeviceName();

        const Version * version = _impl->GetCPPPluginByName(deviceNameLocal).GetVersion();
        versions[deviceNameLocal] = *version;;
    }

    return versions;
}

void Core::SetLogCallback(IErrorListener &listener) const {
    _impl->SetErrorListener(&listener);
}


ExecutableNetwork Core::LoadNetwork(CNNNetwork network, const std::string & deviceName,
                                    const std::map<std::string, std::string> & config) {
    std::map<std::string, std::string> config_ = config;
    std::string deviceName_ = deviceName;

    if (deviceName_.find("HETERO:") == 0) {
        deviceName_ = "HETERO";
        config_["TARGET_FALLBACK"] = deviceName.substr(7);
    } else if (deviceName_.find("MULTI:") == 0) {
        deviceName_ = "MULTI";
        config_[InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES] = deviceName.substr(6);
    } else {
        DeviceIDParser parser(deviceName_);
        deviceName_ = parser.getDeviceName();
        std::string deviceIDLocal = parser.getDeviceID();

        if (!deviceIDLocal.empty()) {
            config_[KEY_DEVICE_ID] = deviceIDLocal;
        }
    }

    std::cerr << "dldt ie_core.cpp LoadNetwork the end" << std::endl;
    return _impl->GetCPPPluginByName(deviceName_).LoadNetwork(network, config_);
}

void Core::AddExtension(IExtensionPtr extension, const std::string & deviceName_) {
    if (deviceName_.find("HETERO") == 0) {
        std::cerr << "dldt ie_core.cpp AddExtension if (deviceName_.find(HETERO) == 0)" << std::endl;
        THROW_IE_EXCEPTION << "HETERO device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("MULTI") == 0) {
        std::cerr << "dldt ie_core.cpp AddExtension if (deviceName_.find(MULTI) == 0)" << std::endl;
        THROW_IE_EXCEPTION << "MULTI device does not support extensions. Please, set extensions directly to fallback devices";
    }

    std::cerr << "dldt ie_core.cpp AddExtension DeviceIDParser parser(deviceName_);" << std::endl;
    DeviceIDParser parser(deviceName_);
    std::cerr << "dldt ie_core.cpp AddExtension std::string deviceName = parser.getDeviceName();" << std::endl;
    std::string deviceName = parser.getDeviceName();

    std::cerr << "dldt AddExtension _impl->GetCPPPluginByName(deviceName).AddExtension(extension);" << std::endl;
    _impl->GetCPPPluginByName(deviceName).AddExtension(extension);
}

ExecutableNetwork Core::ImportNetwork(const std::string &modelFileName, const std::string & deviceName_,
                                      const std::map<std::string, std::string> &config_) {
    if (deviceName_.find("HETERO") == 0) {
        THROW_IE_EXCEPTION << "HETERO device does not support ImportNetwork";
    }
    if (deviceName_.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support ImportNetwork";
    }

    DeviceIDParser parser(deviceName_);
    std::string deviceName = parser.getDeviceName();
    std::string deviceID = parser.getDeviceID();
    auto config = config_;

    // add DEVICE_ID from DEVICE_NAME.# if # is here
    if (!deviceID.empty()) {
        config[KEY_DEVICE_ID] = deviceID;
    }

    return _impl->GetCPPPluginByName(deviceName).ImportNetwork(modelFileName, config);
}

QueryNetworkResult Core::QueryNetwork(const ICNNNetwork &network, const std::string & deviceName,
                                      const std::map<std::string, std::string> & config) const {
    QueryNetworkResult res;
    auto config_ = config;
    std::string deviceName_ = deviceName;

    if (deviceName_.find("MULTI") == 0) {
        THROW_IE_EXCEPTION << "MULTI device does not support QueryNetwork";
    }

    if (deviceName_.find("HETERO:") == 0) {
        deviceName_ = "HETERO";
        config_["TARGET_FALLBACK"] = deviceName.substr(7);
    } else {
        DeviceIDParser parser(deviceName_);
        deviceName_ = parser.getDeviceName();
        std::string deviceIDLocal = parser.getDeviceID();

        if (!deviceIDLocal.empty()) {
            config_[KEY_DEVICE_ID] = deviceIDLocal;
        }
    }

    _impl->GetCPPPluginByName(deviceName_).QueryNetwork(network, config_, res);
    return res;
}

void Core::SetConfig(const std::map<std::string, std::string> & config_, const std::string & deviceName_) {
    // HETERO case
    {
        if (deviceName_.find("HETERO:") == 0) {
            THROW_IE_EXCEPTION << "SetConfig is supported only for HETERO itself (without devices). "
                                  "You can configure the devices with SetConfig before creating the HETERO on top.";
        }

        if (config_.find("TARGET_FALLBACK") != config_.end()) {
            THROW_IE_EXCEPTION << "Please, specify TARGET_FALLBACK to the LoadNetwork directly, "
                                  "as you will need to pass the same TARGET_FALLBACK anyway.";
        }
    }

    // MULTI case
    {
        if (deviceName_.find("MULTI:") == 0) {
            THROW_IE_EXCEPTION << "SetConfig is supported only for MULTI itself (without devices). "
                                  "You can configure the devices with SetConfig before creating the MULTI on top.";
        }

        if (config_.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES) != config_.end()) {
            THROW_IE_EXCEPTION << "Please, specify DEVICE_PRIORITIES to the LoadNetwork directly, "
                                  "as you will need to pass the same DEVICE_PRIORITIES anyway.";
        }
    }

    if (deviceName_.empty()) {
        _impl->SetConfigForPlugins(config_, std::string());
    } else {
        DeviceIDParser parser(deviceName_);
        std::string deviceName = parser.getDeviceName();
        std::string deviceID = parser.getDeviceID();

        auto config = config_;

        // add DEVICE_ID from DEVICE_NAME.# if # is here
        if (!deviceID.empty()) {
            config[KEY_DEVICE_ID] = deviceID;
        }

        _impl->SetConfigForPlugins(config, deviceName);
    }
}

Parameter Core::GetConfig(const std::string & deviceName_, const std::string & name) const {
    // HETERO case
    {
        if (deviceName_.find("HETERO:") == 0) {
            THROW_IE_EXCEPTION << "You can only GetConfig of the HETERO itself (without devices). "
                                  "GetConfig is also possible for the individual devices before creating the HETERO on top.";
        }
    }
    // MULTI case
    {
        if (deviceName_.find("MULTI:") == 0) {
            THROW_IE_EXCEPTION << "You can only GetConfig of the MULTI itself (without devices). "
                                  "GetConfig is also possible for the individual devices before creating the MULTI on top.";
        }
    }

    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();
    std::string deviceID = device.getDeviceID();

    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(deviceName));

    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << deviceName << " does not implement the GetConfig method";
    }

    std::map<std::string, Parameter> config;
    if (!deviceID.empty()) {
        config[KEY_DEVICE_ID] = deviceID;
    }

    return pluginAPIInterface->GetConfig(name, config);
}

Parameter Core::GetMetric(const std::string & deviceName_, const std::string & name) const {
    // HETERO case
    {
        if (deviceName_.find("HETERO:") == 0) {
            THROW_IE_EXCEPTION
                    << "You can get specific metrics with the GetMetric only for the HETERO itself (without devices). "
                       "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    // MULTI case
    {
        if (deviceName_.find("MULTI:") == 0) {
            THROW_IE_EXCEPTION
                    << "You can get specific metrics with the GetMetric only for the MULTI itself (without devices). "
                       "To get individual devices's metrics call GetMetric for each device separately";
        }
    }

    DeviceIDParser device(deviceName_);
    std::string deviceName = device.getDeviceName();
    std::string deviceID = device.getDeviceID();

    auto pluginAPIInterface = getInferencePluginAPIInterface(_impl->GetCPPPluginByName(deviceName));

    if (pluginAPIInterface == nullptr) {
        THROW_IE_EXCEPTION << deviceName << " does not implement the GetMetric method";
    }

    std::map<std::string, Parameter> config;
    if (!deviceID.empty()) {
        config[KEY_DEVICE_ID] = deviceID;
    }

    return pluginAPIInterface->GetMetric(name, config);
}

std::vector<std::string> Core::GetAvailableDevices() const {
    std::vector<std::string> devices;

    std::string propertyName = METRIC_KEY(AVAILABLE_DEVICES);

    for (auto && deviceName : _impl->GetListOfDevicesInRegistry()) {
        Parameter p;
        std::vector<std::string> devicesIDs;

        try {
            p = GetMetric(deviceName, propertyName);
            devicesIDs = p.as<std::vector<std::string> >();
        } catch (details::InferenceEngineException &) {
            // plugin is not created by e.g. invalid env
        } catch (const std::exception & ex) {
            THROW_IE_EXCEPTION << "An exception is thrown while trying to create the " <<
                deviceName << " device and call GetMetric: " << ex.what();
        } catch (...) {
            THROW_IE_EXCEPTION << "Unknown exception is thrown while trying to create the " <<
                deviceName << " device and call GetMetric";
        }

        if (devicesIDs.size() > 1) {
            for (auto && deviceID : devicesIDs) {
                devices.push_back(deviceName + '.' + deviceID);
            }
        } else if (!devicesIDs.empty()) {
            devices.push_back(deviceName);
        }
    }

    return devices;
}

void Core::RegisterPlugin(const std::string & pluginName, const std::string & deviceName) {
    _impl->RegisterPluginByName(pluginName, deviceName);
}

void Core::RegisterPlugins(const std::string & xmlConfigFile) {
    _impl->RegisterPluginsInRegistry(xmlConfigFile);
}

void Core::UnregisterPlugin(const std::string & deviceName_) {
    DeviceIDParser parser(deviceName_);
    std::string deviceName = parser.getDeviceName();

    _impl->UnregisterPluginByName(deviceName);
}

}  // namespace InferenceEngine
