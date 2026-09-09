/**
 * @file Simulator.hpp
 * @author Xein
 * @date 09 Sep 2026
 * @brief This class simulates all of the physics of the project
 */

#pragma once
#include <toast/events/listener.hpp>
#include <toast/log.hpp>

namespace physics {

class Simulator {
public:
	Simulator() {
		TOAST_INFO("Physics", "Simulator created");
		instance = this;
	}

	~Simulator() { instance = nullptr; }

	void tick();

	static void callTick() {
		TOAST_ASSERT(instance, "Physics", "Simulator instance is null; cannot tick");
		instance->tick();
	}

private:
	inline static Simulator* instance = nullptr;
	event::Listener listener;
};

}
