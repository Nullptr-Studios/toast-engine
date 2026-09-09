#include "scripting_test_helpers.hpp"
#include "test_registry.hpp"
#include "toast/scripting/script_runtime.hpp"

#include <cassert>

using namespace toast::tests::scripting_tests;

TOAST_TEST_NAMED("Scripting", "scripting/04_signals", test_scripting_04_signals) {
	luaState();
	auto world_owner = toast::_detail::WorldTestAccess::createWorld();
	auto node = toast::_detail::WorldTestAccess::createNode(*world_owner, "host");

	toast::_detail::WorldTestAccess::attachScript(*node, makeScript(R"lua(
local M = {}
M.value = 0

function M:setup()
    -- forwards_args is intentionally omitted: it defaults to true.
    M.connected = self.test_signal:connect(self:find("root"), "onSignal")
end

function M:onSignal(first, second)
    M.value = first + second
end

function M:emit()
    M.fired = self.test_signal:fire(2, 3)
end

function M:disconnectSignal()
    M.disconnected = self.test_signal:disconnect(self:find("root"), "onSignal")
end

return M
)lua"));

	node->call("setup");
	assert(std::any_cast<bool>(node->scriptRuntime()->getVar("connected")));
	assert(node->test_signal.connections().size() == 1);
	assert(node->test_signal.connections().front().source == signals::ConnectionSource::lua);

	node->call("emit");
	assert(std::any_cast<bool>(node->scriptRuntime()->getVar("fired")));
	assert(std::any_cast<int>(node->scriptRuntime()->getVar("value")) == 5);

	node->call("disconnectSignal");
	assert(std::any_cast<bool>(node->scriptRuntime()->getVar("disconnected")));
	assert(node->test_signal.connections().empty());
}
